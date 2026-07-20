# Used for Data Cleaning and Inspection

library(readr)
library(dplyr)

main <- function() {
  rawData <<- read_csv("processed_by_us.csv", show_col_types = FALSE)
  uniqueTags <<- unique(rawData$tag)

  #empty_rows <<- rawData[rawData$token == "nan", ]
  rawData <<- rawData[rawData$token != "nan", ]
  #write_csv(rawData, "nan_removed_41884rows.csv")
  

  #ids <- sort(unique(rawData$sentence_id))
  #missing_ids <- setdiff(min(ids):max(ids), ids)
  #print(missing_ids)
  
  
  #cat("Total Unique Sentence ID: ", length(unique(rawData$sentence_id)))
  
  differentTag <<- rawData %>%
    group_by(token) %>%
    filter(n_distinct(tag) > 1) %>%
    ungroup() %>%
    arrange(token)
  
  #write_csv(differentTag, "doublecheck.csv")
  
  uniqueTokenTag <<- differentTag %>%
   distinct(token, tag) %>%
    arrange(token)
  
  
 #write_csv(differentTag, "wordsToClean_withGuide.csv")
  
}

main()
