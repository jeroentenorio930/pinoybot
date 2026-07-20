library(readr)

main <- function() {
  rawData <<- read_csv("annotated_dataset_clean.csv", show_col_types = FALSE)
  uniqueTags <<- unique(rawData$tag)
  empty_rows <<- rawData[rawData$token == "nan", ]
  #rawData <<- rawData[rawData$token != "nan", ]
  #write_csv(rawData, "nan_removed_41884rows.csv")
  
  
  
}

main()
