# Core function recipe_title_to_ingredients
# Takes dish title(s) as character strings and returns a data frame containing:
# - The provided dish title
# - The title as recognized and vectorized by the model
# - The predicted ingredients

# prediction_confidence (default 0.5) is the score a final node must exceed to have the ingredient included in the output
# - Making this number larger (up to 1) will show fewer but higher confidence ingredients
# - Smaller numbers (down to 0, not advised as this will return up to 1300 ingredients) will show more but lower confidence ingredients

library(keras)
library(magrittr)
library(readr)
library(stringr)

model <- keras::load_model_hdf5("data/2x2048_dropout20_50epoch_900k.hdf5")
title_word_index <- read_rds("data/title_word_index.rds")
ingr_word_index <- read_rds("data/ingr_word_index.rds")
title_tokenizer <- load_text_tokenizer("data/title_tokenizer")

.sanitize_titles_fun <- function(x){
  x %>%
    str_to_lower() %>%
    str_replace_all('\\Q\"\\E', "") %>%
    str_replace_all('\\Q/\\E', " ") %>%
    str_replace_all('\\\\', " ") %>%
    str_replace_all('\\Q-\\E', " ") %>% #slashes and dashes to spaces
    str_replace_all("\\(|\\)|\\,|\\.|\\'|:|!", "") %>%
    str_replace_all(" +", " ") %>% #make spaces single
    #str_split(pattern = " ") %>%
    return()
}

#' Predict ingredients for a given dish title
#'
#' @param input_titles a list of character strings. required
#' @param prediction_confidence decimal from 0 to 1.
#'
#' @return dataframe with given title, title recognized by model, and predicted ingredients
#' @export
#'
#' @examples
#' recipe_title_to_ingredients(list("best ever carrot cake", "mom's pineapple zucchini bread"))
#' recipe_title_to_ingredients(input_titles = list("meatless stew","meatless beef stew","meatless meatlover's stew","meatless meatlovers beef stew"), prediction_confidence = 0.5)
#'
#' recipe_title_to_ingredients(input_titles = list("meatless stew"), prediction_confidence = 0.7)
#' recipe_title_to_ingredients(input_titles = list("meatless stew"), prediction_confidence = 0.5)
#' recipe_title_to_ingredients(input_titles = list("meatless stew"), prediction_confidence = 0.3)
#'
recipe_title_to_ingredients <-
  function(input_titles,
           prediction_confidence = 0.5){

    clean_dish <- input_titles %>%
      .sanitize_titles_fun()

    vec_dish <- texts_to_sequences(title_tokenizer, clean_dish) %>%
      pad_sequences(maxlen = 10)

    #print(vec_dish)



    predictions <- model %>% predict(vec_dish)

    predictions_binary <- predictions >= prediction_confidence

    vectored_title <- vector(length = nrow(predictions), mode = "character")
    ingrs <- vector(length = nrow(predictions), mode = "character")

    for(i in 1:nrow(predictions)){
      if(ingr_word_index[predictions_binary[i,]] %>% names() %>% length() > 0){
        ingrs[i] = ingr_word_index[predictions_binary[i,]] %>% names() %>% str_c(collapse = ", ")
      }
      if(title_word_index[vec_dish[i,]] %>% names() %>% length() > 0){
        vectored_title[i] = title_word_index[vec_dish[i,]] %>% names() %>% str_c(collapse = " ")
      }
    }



    predicted_df <- data.frame(title = as.character(input_titles), recognized_title = vectored_title, ingrs = ingrs)

    return(predicted_df)
  }
