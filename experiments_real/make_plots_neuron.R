options(width=160)

library(ggplot2)
library(tidyverse)
library(kableExtra)
library(patchwork)
library(purrr)



results_dir_root <- "C:/Users/liang/Documents/GitHub/fusion/experiments_real/results/Neuron/"
tab.dir <- "C:/Users/liang/Documents/GitHub/fusion/results/tables/"
fig.dir <- "C:/Users/liang/Documents/GitHub/fusion/results/figures/"

method_mapping <- c(
  "modality_1" = "LFP",
  "modality_2" = "Spike",
  "late_fusion" = "Late Fusion",
  "early_fusion" = "Early Fusion",
  "greedy_ensemble" = "Meta Fusion",
  "simple_average" = "Simple Avg.",
  "weighted_average" = "Weighted Avg.",
  "majority_voting" = "Majority Vote",
  "weighted_voting" = "Weighted Vote",
  "meta_learner" = "Stacking",
  "indep_best_single" = "Best Single (ind.)",
  "best_single" = "Best Single"
)


all_rats <- c("Barat", "Buchanan", "Stella", "Superchris", "Mitt")
extractor_mapping <- c("Barat"="encoder", "Buchanan"="encoder", "Stella"="encoder",
                       "Superchris"="PCA", "Mitt"="PCA")



#--------------------------#
#----Processing results----#
#--------------------------#
all_data <- list()
for (rat_name in all_rats) {
  results_dir <- file.path(results_dir_root, rat_name)
  file_list <- list.files(path = results_dir, full.names = TRUE)
  for (file in file_list) {
    df <- read_csv(file, col_names = TRUE, show_col_types = FALSE)
    # The dataframe already contains 'rat_name' and 'extractor', so nothing else needed
    all_data[[paste0(rat_name, "_", basename(file))]] <- df
  }
}

# Combine all data and keep only rows with correct extractor per rat
all_data_df <- bind_rows(all_data) %>%
  fill(cohort_pairs, .direction = "downup") %>%
  filter(extractor == recode(rat_name, !!!extractor_mapping))

non_cohort_data <- all_data_df %>%
  filter(!grepl("cohort", Method)) %>%
  mutate(Test_metric = as.numeric(Test_metric)) %>%
  group_by(rat_name, extractor, Method) %>%
  summarise(
    mean_metric = mean(Test_metric, na.rm = TRUE),
    se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
    count = n(),
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
  formatted <- sprintf("%.3f (%.3f)", acc, se)
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
  fusion_methods <- c("modality_1", "modality_2", "early_fusion", "late_fusion", "greedy_ensemble")
  non_cohort_data %>%
    filter(Method %in% fusion_methods) %>%
    mutate(Method = factor(Method, levels = fusion_methods),
           Method = recode(Method, !!!method_mapping)) %>%
    select(rat_name, Method, `Mean Acc`, `SE Acc`) %>%
    group_by(rat_name) %>%
    mutate(
      best_acc = max(`Mean Acc`, na.rm=TRUE),
      best_se = `SE Acc`[which.max(`Mean Acc`)]
    ) %>%
    ungroup() %>%
    rowwise() %>%
    mutate(entry = format_entry(`Mean Acc`, `SE Acc`, best_acc, best_se)) %>%
    ungroup() %>%
    select(Method, rat_name, entry) %>%
    pivot_wider(id_cols = Method, names_from = rat_name, values_from = entry) %>%
    arrange(match(Method, method_mapping[c("modality_1", "modality_2", "early_fusion", "late_fusion", "greedy_ensemble")])) %>%
    rename(` ` = Method) %>%
    kbl(format="latex", booktabs=TRUE, escape=FALSE, caption = NULL,
        align=c("l", rep("c", ncol(.) - 1)), linesep = linesep(c(2,3))) %>%
    save_kable(file=sprintf("%s/%s.tex", save_dir, fname), keep_tex=TRUE, self_contained=FALSE)
}

make_table_fusion(non_cohort_data, method_mapping, "Neuron_odor", tab.dir)



make_table_ensemble <- function(non_cohort_data, method_mapping, fname, save_dir) {
  ensemble_methods <- c("indep_best_single", "best_single", "greedy_ensemble", 
                        "meta_learner", "simple_average", "weighted_average", 
                        "majority_voting", "weighted_voting")
  non_cohort_data %>%
    filter(Method %in% ensemble_methods) %>%
    mutate(Method = factor(Method, levels = ensemble_methods),
           Method = recode(Method, !!!method_mapping)) %>%
    select(rat_name, Method, `Mean Acc`, `SE Acc`) %>%
    group_by(rat_name) %>%
    mutate(
      best_acc = max(`Mean Acc`, na.rm=TRUE),
      best_se = `SE Acc`[which.max(`Mean Acc`)]
    ) %>%
    ungroup() %>%
    rowwise() %>%
    mutate(entry = format_entry(`Mean Acc`, `SE Acc`, best_acc, best_se)) %>%
    ungroup() %>%
    select(Method, rat_name, entry) %>%
    pivot_wider(id_cols = Method, names_from = rat_name, values_from = entry) %>%
    arrange(match(Method, method_mapping[ensemble_methods])) %>%
    rename(` ` = Method) %>%
    kbl(format="latex", booktabs=TRUE, escape=FALSE, caption = NULL,
        align=c("l", rep("c", ncol(.) - 1)), linesep = linesep(c(3,3,2))) %>%
    save_kable(file=sprintf("%s/%s.tex", save_dir, fname), keep_tex=TRUE, self_contained=FALSE)
}

make_table_ensemble(non_cohort_data, method_mapping, "Neuron_odor_ensemble", tab.dir)



plot_fusion_one_rat <- function(df, method_mapping, rat, y_limits = c(0,1), plot_full) {
  color_map <- c(
    "LFP"          = "#66CCFF",
    "Spike"        = "#6691FF",
    "Early Fusion" = "#ffe000",
    "Late Fusion"  = "#FF9900",
    "Meta Fusion"  = "#CC3333"
  )
  if (plot_full){
    fusion_methods <- c("modality_1", "modality_2", "early_fusion", "late_fusion", "greedy_ensemble")
  }else{
    fusion_methods <- c("modality_2", "late_fusion", "greedy_ensemble")
  }
  
  # Filter just this rat
  df_rat <- df %>%
    filter(rat_name == rat, Method %in% fusion_methods) %>%
    mutate(
      Method = factor(Method, levels = fusion_methods),
      Method = recode(Method, !!!method_mapping)
    )
  
  ggplot(df_rat, aes(x = Method, y = `Mean Acc`, fill = Method)) +
    geom_col() +
    geom_errorbar(aes(
      ymin = `Mean Acc` - `SE Acc`,
      ymax = `Mean Acc` + `SE Acc`
    ), width = 0.2) +
    scale_fill_manual(values = color_map) +
    coord_cartesian(ylim=y_limits) +
    labs(title = rat, y = NULL, x = NULL, fill = NULL) +
    theme_bw(base_size = 12) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank()
    )
}

plot_fusion_all_rats <- function(non_cohort_data, method_mapping, 
                                 fname, save_dir, plot_full) {
  
  plot_height=3
  if (plot_full){
    custom_limits <- tribble(
      ~rats, ~y_min, ~y_max,
      "Barat", 0.45, 0.65,
      "Buchanan", 0.45, 0.8,
      "Mitt", 0.3, 0.65,
      "Stella", 0.5, 0.7,
      "Superchris", 0.35, 0.8
    )
    plot_width=9
  }
  else{
    custom_limits <- tribble(
      ~rats, ~y_min, ~y_max,
      "Barat", 0.5, 0.65,
      "Buchanan", 0.65, 0.8,
      "Mitt", 0.5, 0.65,
      "Stella", 0.55, 0.7,
      "Superchris", 0.6, 0.8
    )
    plot_width=8
  }
  rat_plots <- custom_limits %>%
    rowwise() %>%
    mutate(p = list(
      plot_fusion_one_rat(
        df = non_cohort_data,
        method_mapping = method_mapping,
        rat = rats,
        y_limits = c(y_min, y_max),
        plot_full = plot_full
      )
    )) %>%
    pull(p)
  
  custom_lyt <- "AAABBBCCCDDDEEE
                 AAABBBCCCDDDEEE
                 AAABBBCCCDDDEEE
                 AAABBBCCCDDDEEE
                 AAABBBCCCDDDEEE
                 #####ZZZZZ#####"
  
  combined_plot <- reduce(rat_plots, `+`) + 
    guide_area() + 
    plot_layout(nrow = 1, guides = "collect", design = custom_lyt) & 
    theme(
    plot.title         = element_text(hjust = 0.5, size = 12),# center rat names
    axis.text.y        = element_text(size = 10),
    legend.text        = element_text(size = 12),      # bigger labels in legend
    #legend.key.size    = grid::unit(0.8, "cm")         # bigger legend boxes
    )
  
  ggsave(file.path(save_dir, paste0(fname, ifelse(plot_full, "_full", ""), ".pdf")),
         plot = combined_plot, width = plot_width, height = plot_height)
}

plot_fusion_all_rats(non_cohort_data, method_mapping, "Neuron_odor", fig.dir, plot_full=FALSE)
plot_fusion_all_rats(non_cohort_data, method_mapping, "Neuron_odor", fig.dir, plot_full=TRUE)

