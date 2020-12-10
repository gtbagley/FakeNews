##Cleaning fake news data, modeling using xgboost, and generating predictions

##The first portion of the code is basically that of MJHeaton, used for cleaning generating necessary variables for language processing
## Libraries
library(tidyverse)
library(tidytext)
library(stopwords)
sessionInfo()
## Read in the Data
fakeNews.test <- read.csv("./fake-news/test.csv")
fakeNews.train <- read.csv("./fake-news/train.csv")

fakeNews <- bind_rows(train=fakeNews.train, test=fakeNews.test,
                      .id="Set")

################################
## Create a language variable ##
################################

## Determine which language each article is in
fakeNews <- fakeNews %>%
  mutate(language=textcat::textcat(text))

count(fakeNews$language) %>% arrange(desc(freq))
## Combine some languages into same category
fakeNews <- fakeNews %>%
  mutate(language=fct_collapse(language, 
                               english=c("english", "middle_frisian", "scots",
                                         "scots_gaelic", "breton", "frisian",
                                         "manx", "catalan"),
                               russian=c("russian-koi8_r", "russian-iso8859_5",
                                         "russian-windows1251")))

## Lump together other languages
fakeNews <- fakeNews %>% 
  mutate(language=fct_explicit_na(language, na_level="Missing")) %>%
  mutate(language=fct_lump(language, n=6))

count(fakeNews$language) %>% arrange(desc(freq))

############################################
## Calculate df-idf for most common words ##
## not including stop words               ##
############################################

## Create a set of stop words
sw <- bind_rows(get_stopwords(language="en"), #English
                get_stopwords(language="ru"), #Russian
                get_stopwords(language="es"), #Spanish
                get_stopwords(language="de"), #German
                get_stopwords(language="fr")) #French
sw <- sw %>%
  bind_rows(., data.frame(word="это", lexicon="snowball"))

## tidytext format
tidyNews <- fakeNews %>%
  unnest_tokens(tbl=., output=word, input=text)

## Count of words in each article
news.wc <-  tidyNews %>%
  anti_join(sw) %>% 
  count(id, word, sort=TRUE)


## Number of non-stop words per article
all.wc <- news.wc %>% 
  group_by(id) %>% 
  summarize(total = sum(n))

## Join back to original df and calculate term frequency
news.wc <- left_join(news.wc, all.wc) %>%
  left_join(x=., y=fakeNews %>% select(id, title))
news.wc <- news.wc %>% mutate(tf=n/total)
a.doc <- sample(news.wc$title,1)
ggplot(data=(news.wc %>% filter(title==a.doc)), aes(tf)) +
  geom_histogram() + ggtitle(label=a.doc)

## Find the tf-idf for the most common p% of words
word.count <- news.wc %>%
  count(word, sort=TRUE) %>%
  mutate(cumpct=cumsum(n)/sum(n))
ggplot(data=word.count, aes(x=1:nrow(word.count), y=cumpct)) + 
  geom_line()
top.words <- word.count %>%
  filter(cumpct<0.75)

news.wc.top <- news.wc %>% filter(word%in%top.words$word) %>%
  bind_tf_idf(word, id, n)
true.doc <- sample(fakeNews %>% filter(label==0) %>% pull(title),1)
fake.doc <- sample(fakeNews %>% filter(label==1) %>% pull(title),1)


############################################
## Merge back with original fakeNews data ##
############################################

## Convert from "long" data format to "wide" data format
## so that word tfidf become explanatory variables
names(news.wc.top)[1] <- "Id"
news.tfidf <- news.wc.top %>%
  pivot_wider(id_cols=Id,
              names_from=word,
              values_from=tf_idf)

## Fix NA's to zero
news.tfidf <- news.tfidf %>%
  replace(is.na(.), 0)

## Merge back with fakeNews data
names(fakeNews)[c(2,6)] <- c("Id", "isFake")
fakeNews.tfidf <- left_join(fakeNews, news.tfidf, by="Id")

## Remaining articles with NAs all have missing text so should get 0 tfidf
fakeNews.clean <- fakeNews.tfidf %>%
  select(-isFake, -title.x, -author.x, -text.x) %>% 
  replace(is.na(.), 0) %>% 
  left_join(fakeNews.tfidf %>% select(Id, isFake, title.x, author.x, text.x),., by="Id")

## Compare distributions of tfidf for different words
a.word <- sample(names(fakeNews.clean)[-c(1:7)], 1)
z <- as.name(a.word)
class(z)
sub.df <- fakeNews.clean[,c(a.word, "isFake")]
names(sub.df) <- c("x", "isFake")
ggplot(data=sub.df %>% filter(x>0), mapping=aes(x=x, color=as.factor(isFake))) +
  geom_density() + ggtitle(label=a.word)

## From here on the code is basically all mine

# Create variable for ratio of caps letters to all letters
library(plyr)
caps_ratio <- ldply(str_match_all(fakeNews.clean$text.x,"[A-Z]"),length)/nchar(fakeNews.clean$text.x)
fakeNews.clean$caps_ratio <- caps_ratio


# change empty title.x level to "No Title"
fakeNews.clean$title.x[fakeNews.clean$title.x == ""] <- "No Title"

# Combine nan and -NO AUTHOR- levels for author into Unknown, 
fakeNews.clean <- fakeNews.clean %>% mutate(author.x=fct_collapse(author.x, Unknown=c("-NO AUTHOR-", "nan")))

# Create indicator variables for ureliable author, no title, 
x <- fakeNews.clean$author.x[fakeNews.clean$label.x == 1]
fakeNews.clean$unreliable_author <- fakeNews.clean$author.x %in% x
fakeNews.clean$no_title <- as.numeric(fakeNews.clean$title.x == "No Title")
fakeNews.clean$unknown_author <- as.numeric(fakeNews.clean$author.x == "Unknown")

#Code below copied from other file, so I change the name
CleanFakeNews <- fakeNews.clean

# split into test and training
CleanFakeNews.train <- CleanFakeNews %>% filter(Set == 'train')
CleanFakeNews.test <- CleanFakeNews %>% filter(Set == 'test')

# create model
boost1 <- train(form=as.factor(isFake)~.,
                data=fn.train %>% select(-Id, -Set),
                method="xgbTree",
                trControl=trainControl(method="cv",
                                       number=5),
                tuneGrid = expand.grid(nrounds=100, # Boosting Iterations
                                       max_depth=3, #Max Tree Depth
                                       eta=0.4, #(Shrinkage)
                                       gamma=1,
                                       colsample_bytree=1,# (Subsample Ratio of Columns)
                                       min_child_weight=1,# (Minimum Sum of Instance Weight)
                                       subsample=1)# (Subsample Percentage)0)
)
beepr::beep(sound=5)

preds <- predict(boost1, newdata=fn.test)

predframe <- data.frame(id=fn.test$Id, label=preds)

write.csv(x=predframe,file="./fake-news/fakenews.csv", row.names=FALSE)
