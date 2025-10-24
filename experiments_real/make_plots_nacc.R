options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)
library(stringr)
library(purrr)



results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_real/results/NACC_fine_grained/"
tab.dir <- "C:/Users/liang/Documents/GitHub/fusion/results/tables/"
fig.dir <- "C:/Users/liang/Documents/GitHub/fusion/results/figures/"

method_mapping <- c(
  "modality_1" = "Patient Profile",
  "modality_2" = "Behavioral Assmt.",
  "modality_3" = "Clinical Eval.",
  "modality_4" = "MRI",
  "early_fusion" = "Early Fusion",
  "late_fusion" = "Late Fusion",
  "greedy_ensemble" = "Meta Fusion",
  "simple_average" = "Simple Avg.",
  "weighted_average" = "Weighted Avg.",
  "majority_voting" = "Majority Vote",
  "weighted_voting" = "Weighted Vote",
  "meta_learner" = "Stacking",
  "indep_best_single" = "Best Single (ind.)",
  "best_single" = "Best Single"
)



#--------------------------#
#----Processing results----#
#--------------------------#
file_list <- list.files(path = results_dir, full.names = TRUE)
data_list <- list()
for (file in file_list) {
  df <- read_csv(file, col_names = TRUE, show_col_types = FALSE)
  data_list[[file]] <- df
}

all_data <- bind_rows(data_list) %>%
            fill(cohort_pairs, .direction = "downup")

# Non-cohort methods
non_cohort_data <- all_data %>%
  filter(!grepl("cohort", Method), extractor=='encoder') %>%
  mutate(Test_metric = as.numeric(Test_metric)) %>%
  group_by(Method) %>%
  summarise(
    mean_metric = mean(Test_metric, na.rm = TRUE),
    se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),  # Calculate SE
    count = n(),  # Count the number of repetitions
    .groups = "drop"
  ) %>%
  rename(
    "Mean Acc" = mean_metric,
    "SE Acc" = se_metric
  )




#--------------------------#
#-----Helper functions-----#
#--------------------------#
format_entry <- function(acc, se, best_acc, best_se) {
  formatted <- sprintf("%.4f (%.4f)", acc, se)
  if (!is.na(acc) && (acc >= best_acc - best_se)) {
    formatted <- sprintf("\\textbf{%s}", formatted)
  }
  formatted
}

# Control line separation
linesep<-function(x,y=character()){
  if(!length(x))
    return(y)
  linesep(x[-length(x)], c(rep('',x[length(x)]-1),'\\addlinespace',y))  
}




#--------------------------#
#------Tables & Plots------#
#--------------------------#
make_table_fusion <- function(non_cohort_data, method_mapping, fname, save_dir) {
  fusion_methods <- c("modality_1", "modality_2", "modality_3", "modality_4", "early_fusion", "late_fusion", "greedy_ensemble")
  
  non_cohort_data %>%
    filter(Method %in% fusion_methods) %>%
    mutate(Method = factor(Method, levels = fusion_methods),
           Method = recode(Method, !!!method_mapping)) %>%
    select(Method, `Mean Acc`, `SE Acc`) %>%
    mutate(
      best_acc = max(`Mean Acc`, na.rm=TRUE),
      best_se = `SE Acc`[which.max(`Mean Acc`)]
    ) %>%
    rowwise() %>%
    mutate(entry = format_entry(`Mean Acc`, `SE Acc`, best_acc, best_se)) %>%
    ungroup() %>%
    select(Method, entry) %>%
    arrange(match(Method, method_mapping[fusion_methods])) %>%
    rename(` ` = Method) %>%
    kbl(format="latex", booktabs=TRUE, escape=FALSE, caption = NULL,
        col.names = c("", "Mean Acc (SE)"),
        align=c("l", "c"), linesep = linesep(c(4,3))) %>%
    save_kable(file=sprintf("%s/%s.tex", save_dir, fname), keep_tex=TRUE, self_contained=FALSE)
}

make_table_fusion(non_cohort_data, method_mapping, "NACC", tab.dir)



make_table_ensemble <- function(non_cohort_data, method_mapping, fname, save_dir) {
  ensemble_methods <- c("indep_best_single", "best_single", "greedy_ensemble", 
                        "meta_learner", "simple_average", "weighted_average", 
                        "majority_voting", "weighted_voting")
  non_cohort_data %>%
    filter(Method %in% ensemble_methods) %>%
    mutate(Method = factor(Method, levels = ensemble_methods),
           Method = recode(Method, !!!method_mapping)) %>%
    select(Method, `Mean Acc`, `SE Acc`) %>%
    mutate(
      best_acc = max(`Mean Acc`, na.rm=TRUE),
      best_se = `SE Acc`[which.max(`Mean Acc`)]
    ) %>%
    ungroup() %>%
    rowwise() %>%
    mutate(entry = format_entry(`Mean Acc`, `SE Acc`, best_acc, best_se)) %>%
    ungroup() %>%
    select(Method, entry) %>%
    arrange(match(Method, method_mapping[ensemble_methods])) %>%
    rename(` ` = Method) %>%
    kbl(format="latex", booktabs=TRUE, escape=FALSE, caption = NULL,
        col.names = c("", "Mean Acc (SE)"),
        align=c("l", "c"), linesep = linesep(c(3,3,2))) %>%
    save_kable(file=sprintf("%s/%s.tex", save_dir, fname), keep_tex=TRUE, self_contained=FALSE)
}

make_table_ensemble(non_cohort_data, method_mapping, "NACC_ensemble", tab.dir)



plot_fusion_nacc <- function(df, method_mapping, fname, save_dir) {
  color_map <- c(
    "Patient Profile"       = "#66CCFF",
    "Behavioral Assmt." = "#6691FF",
    "Clinical Eval."   = "#66FF66",
    "MRI"                   = "#33CC00",
    " "                     = "white",      # Create a gap
    "Early Fusion"          = "#ffe000",
    "Late Fusion"           = "#FF9900",
    "Meta Fusion"           = "#CC3333"
  )
  
  fusion_methods <- c("modality_1", "modality_2", "modality_3", "modality_4", " ",
                      "early_fusion", "late_fusion", "greedy_ensemble")
  
  df_fusion <- df %>%
    filter(Method %in% fusion_methods)
  
  # Add a dummy row for the gap
  gap_row <- df_fusion[1, ]
  gap_row$Method <- " "
  gap_row$`Mean Acc` <- 0
  gap_row$`SE Acc` <- 0
  
  df_fusion <- rbind(df_fusion, gap_row) %>%
    mutate(
      Method = factor(Method, levels = fusion_methods),
      Method = recode(Method, !!!method_mapping)
    )
  
  p<- ggplot(df_fusion, aes(x = Method, y = `Mean Acc`, fill = Method)) +
      geom_col() +
      geom_errorbar(aes(
        ymin = `Mean Acc` - `SE Acc`,
        ymax = `Mean Acc` + `SE Acc`
      ), width = 0.2) +
      scale_fill_manual(values = color_map) +
      coord_cartesian(ylim = c(0.65, 0.80)) +
      labs(title = NULL, y = "Average Accuracy", x = NULL, fill = NULL) +
      theme_bw(base_size = 12) +
      theme(
        legend.position = "right",
        legend.text = element_text(size = 13),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size= 14, margin = margin(r = 10)),
        axis.text.x = element_blank(),
        plot.title = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank()
      )
  
  ggsave(file.path(save_dir, paste0(fname, ".pdf")),
         plot = p, width = 7, height = 3)
}

plot_fusion_nacc(non_cohort_data, method_mapping, "NACC", fig.dir)
