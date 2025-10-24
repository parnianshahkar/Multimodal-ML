options(width=160)

library(ggplot2)
library(tidyverse)
library(kableExtra)
library(purrr)
library(dplyr)
library(tidyr)
library(patchwork)
library(viridis)

#-------------------------------#
#--------- Plot Utils ----------# 
#-------------------------------#
process_experiment <- function(exp_name, results_dir, setting, ratio_filter = NULL) {
  full_path <- file.path(results_dir, exp_name)
  
  file_list <- list.files(path = full_path, full.names = TRUE)
  
  # Filter files based on ratio if specified
  if (!is.null(ratio_filter)) {
    ratio_pattern <- paste0("noise_ratio", ratio_filter)
    file_list <- file_list[grep(ratio_pattern, file_list)]
  }
  
  data_list <- lapply(file_list, read_csv, col_names = TRUE, show_col_types = FALSE)
  
  all_data <- bind_rows(data_list)
  
  non_cohort_data <- all_data %>%
    filter(!grepl("cohort", Method)) %>%
    mutate(Test_metric = as.numeric(Test_metric),
           Setting = setting)
  
  return(non_cohort_data)
}

# Helper function to format entries
format_entry <- function(mse, se, best_mse, best_se) {
  formatted <- sprintf("%.2f (%.2f)", mse, se)
  if (!is.na(mse) && (mse <= best_mse + best_se)) {
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

# Function to make fusion table
make_table_fusion <- function(non_cohort_data, method_mapping, fname, save_dir, col_names) {
  fusion_methods <- c("modality_1", "modality_2", "early_fusion", "late_fusion", "coop", "greedy_ensemble")
  
  fusion_results <- non_cohort_data %>%
    filter(Method %in% fusion_methods) %>%
    group_by(Setting, Method) %>%
    summarise(
      mean_metric = mean(Test_metric, na.rm = TRUE),
      se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    group_by(Setting) %>%
    mutate(
      best_mse = min(mean_metric, na.rm = TRUE),
      best_se = se_metric[which.min(mean_metric)]
    ) %>%
    ungroup() %>%
    rowwise() %>%
    mutate(Entry = format_entry(mean_metric, se_metric, best_mse, best_se)) %>%
    select(Setting, Method, Entry) %>%
    pivot_wider(names_from = Setting, values_from = Entry) %>%
    mutate(Method = recode(Method, !!!method_mapping)) %>%
    arrange(match(Method, method_mapping[fusion_methods]))
  
  kable(fusion_results, format = "latex", booktabs = TRUE, escape = FALSE,caption = NULL,
        col.names = col_names,
        align = c("l", "c", "c", "c"),
        linesep = linesep(c(2,4))) %>%
    save_kable(file = file.path(save_dir, paste0(fname, "_fusion.tex")), keep_tex = TRUE)
  
  return(fusion_results)
}

# Function to make ensemble table
make_table_ensemble <- function(non_cohort_data, method_mapping, fname, save_dir, col_names) {
  ensemble_methods <- c("indep_best_single", "best_single", "greedy_ensemble",
                        "meta_learner", "simple_average", "weighted_average")

  ensemble_results <- non_cohort_data %>%
    filter(Method %in% ensemble_methods) %>%
    group_by(Setting, Method) %>%
    summarise(
      mean_metric = mean(Test_metric, na.rm = TRUE),
      se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    group_by(Setting) %>%
    mutate(
      best_mse = min(mean_metric, na.rm = TRUE),
      best_se = se_metric[which.min(mean_metric)]
    ) %>%
    ungroup() %>%
    rowwise() %>%
    mutate(Entry = format_entry(mean_metric, se_metric, best_mse, best_se)) %>%
    select(Setting, Method, Entry) %>%
    pivot_wider(names_from = Setting, values_from = Entry) %>%
    mutate(Method = recode(Method, !!!method_mapping)) %>%
    arrange(match(Method, method_mapping[ensemble_methods]))

  kable(ensemble_results, format = "latex", booktabs = TRUE, escape = FALSE, caption = NULL,
        col.names = col_names,
        align = c("l", "c", "c", "c"),
        linesep = linesep(c(3,3))) %>%
    save_kable(file = file.path(save_dir, paste0(fname, "_ensemble.tex")), keep_tex = TRUE)

  return(ensemble_results)
}

# make_table_ensemble <- function(non_cohort_data, method_mapping, fname, save_dir, col_names) {
#   ensemble_methods <- c("best_single", "greedy_ensemble", 
#                         "meta_learner", "simple_average", "weighted_average")
#   
#   all_methods <- c(ensemble_methods, paste0("indep_", ensemble_methods))
#   
#   ensemble_results <- non_cohort_data %>%
#     filter(Method %in% all_methods) %>%
#     group_by(Setting, Method) %>%
#     summarise(
#       mean_metric = mean(Test_metric, na.rm = TRUE),
#       se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
#       .groups = "drop"
#     ) %>%
#     group_by(Setting) %>%
#     mutate(
#       best_mse = min(mean_metric, na.rm = TRUE),
#       best_se = se_metric[which.min(mean_metric)]
#     ) %>%
#     ungroup() %>%
#     rowwise() %>%
#     mutate(Entry = format_entry(mean_metric, se_metric, best_mse, best_se)) %>%
#     select(Setting, Method, Entry) %>%
#     mutate(ML_Type = ifelse(grepl("^indep_", Method), "No AML", "AML"),
#            Method = gsub("^indep_", "", Method)) %>%
#     pivot_wider(names_from = c(Setting, ML_Type), values_from = Entry) %>%
#     mutate(Method = recode(Method, !!!method_mapping)) %>%
#     arrange(match(Method, method_mapping[ensemble_methods]))
#   
#   # Create column names for AML and No AML
#   aml_col_names <- c("", rep(c("AML", "No AML"), length(col_names) - 1))
#   names(aml_col_names) <- c("", rep(col_names[-1], each = 2))
#   
#   kable(ensemble_results, format = "latex", booktabs = TRUE, escape = FALSE, caption = NULL,
#         col.names = aml_col_names,
#         align = c("l", rep("c", 2 * (length(col_names) - 1))),
#         linesep = linesep(c(2,3))) %>%
#     add_header_above(c(" " = 1, setNames(rep(2, length(col_names) - 1), col_names[-1]))) %>%
#     save_kable(file = file.path(save_dir, paste0(fname, "_ensemble.tex")), keep_tex = TRUE)
#   
#   return(ensemble_results)
# }

#-------------------------------#
#-------- Early settings -------# 
#-------------------------------#
results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_synthetic/results/"
save_dir <- "C:/Users/liang/Documents/GitHub/fusion/results/tables/"

method_mapping <- c(
  "modality_1" = "Modality 1",
  "modality_2" = "Modality 2",
  "early_fusion" = "Early Fusion",
  "late_fusion" = "Late Fusion",
  "coop" = "Coop",
  "greedy_ensemble" = "Meta Fusion",
  "simple_average" = "Simple Avg.",
  "weighted_average" = "Weighted Avg.",
  "meta_learner" = "Stacking",
  "best_single" = "Best Single",
  "indep_best_single" = "Best Single (ind.)"
)

# Process both experiments
linear_data <- process_experiment("regression_linear_early", results_dir, "Setting 1.1")
quadratic_data <- process_experiment("regression_quadratic_early", results_dir, "Setting 1.2", ratio_filter = 0.1)
noisy_quadratic_data <- process_experiment("regression_quadratic_early", results_dir, "Setting 1.3", ratio_filter = 0.5)
combined_data <- bind_rows(linear_data, quadratic_data, noisy_quadratic_data)

# Generate tables
col_names <- c("", "Setting 1.1", "Setting 1.2", "Setting 1.3")
fusion_table <- make_table_fusion(combined_data, method_mapping, "regression_early", save_dir, col_names)
ensemble_table <- make_table_ensemble(combined_data, method_mapping, "regression_early", save_dir, col_names)



#-------------------------------#
#-------- Late settings --------# 
#-------------------------------#
results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_synthetic/results/"
save_dir <- "C:/Users/liang/Documents/GitHub/fusion/results/tables/"

method_mapping <- c(
  "modality_1" = "Modality 1",
  "modality_2" = "Modality 2",
  "early_fusion" = "Early Fusion",
  "late_fusion" = "Late Fusion",
  "coop" = "Coop",
  "greedy_ensemble" = "Meta Fusion",
  "simple_average" = "Simple Avg.",
  "weighted_average" = "Weighted Avg.",
  "meta_learner" = "Stacking",
  "best_single" = "Best Single",
  "indep_best_single" = "Best Single (ind.)"
)

# Process both experiments
linear_data <- process_experiment("regression_linear_late", results_dir, "Setting 2.1")
quadratic_data <- process_experiment("regression_quadratic_late", results_dir, "Setting 2.2", ratio_filter = 0.3)
noisy_quadratic_data <- process_experiment("regression_quadratic_late", results_dir, "Setting 2.3", ratio_filter = 0.5)
combined_data <- bind_rows(linear_data, quadratic_data, noisy_quadratic_data)

# Generate tables
col_names <- c("", "Setting 2.1", "Setting 2.2", "Setting 2.3")
fusion_table <- make_table_fusion(combined_data, method_mapping, "regression_late", save_dir, col_names)
ensemble_table <- make_table_ensemble(combined_data, method_mapping, "regression_late", save_dir, col_names)



#-------------------------------#
#-------- Ensemble Plot---------# 
#-------------------------------#
plot_ensemble_one_setting <- function(df, method_mapping, setting, y_limits = c(0,1)) {
  color_map <- c(
    "Best Single (ind.)" = "#66CCFF",
    "Best Single"        = "#6691FF",
    "Meta Fusion"        = "#CC3333",
    "Stacking"           = "#FF9900",
    "Weighted Avg."      = "#ffe000"
  )
  
  ensemble_methods <- c("indep_best_single", "best_single", "greedy_ensemble",
                        "meta_learner", "weighted_average")
  
  df_setting <- df %>%
    filter(Setting == setting, Method %in% ensemble_methods) %>%
    mutate(
      Method = factor(Method, levels = ensemble_methods),
      Method = recode(Method, !!!method_mapping)
    )
  
  ggplot(df_setting, aes(x = Method, y = mean_metric, fill = Method)) +
    geom_col() +
    geom_errorbar(aes(
      ymin = mean_metric - se_metric,
      ymax = mean_metric + se_metric
    ), width = 0.2) +
    scale_fill_manual(values = color_map) +
    coord_cartesian(ylim = y_limits) +
    labs(title = setting, y = NULL, x = NULL, fill = NULL) +
    theme_bw(base_size = 12) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank()
    )
}

make_ensemble_plot <- function(non_cohort_data, method_mapping, fname, save_dir, col_names) {
  ensemble_results <- non_cohort_data %>%
    filter(Method %in% c("indep_best_single", "best_single", "greedy_ensemble",
                         "meta_learner", "weighted_average")) %>%
    group_by(Setting, Method) %>%
    summarise(
      mean_metric = mean(Test_metric, na.rm = TRUE),
      se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  # Define custom y-axis limits for each setting
  custom_limits <- tribble(
    ~Setting, ~y_min, ~y_max,
    "Setting 2.1", 2.3, 2.8,
    "Setting 2.2", 50.0, 63,
    "Setting 2.3", 51, 67
  )
  
  setting_plots <- custom_limits %>%
    rowwise() %>%
    mutate(p = list(
      plot_ensemble_one_setting(
        df = ensemble_results,
        method_mapping = method_mapping,
        setting = Setting,
        y_limits = c(y_min, y_max)
      )
    )) %>%
    pull(p)
  
  combined_plot <- reduce(setting_plots, `+`) + 
    plot_layout(nrow = 1, guides = "collect") & 
    theme(legend.position = "right",
          plot.title         = element_text(hjust = 0.5, size = 12),# center rat names
          axis.text.y        = element_text(size = 10),
          legend.text        = element_text(size = 12),      # bigger labels in legend
          #legend.key.size    = grid::unit(0.8, "cm")         # bigger legend boxes
          )
  
  ggsave(file.path(save_dir, paste0(fname, "_ensemble_plot.pdf")),
         plot = combined_plot, width = 8.5, height = 2.5)
  
  return(ensemble_results)
}

save_dir <- "C:/Users/liang/Documents/GitHub/fusion/results/figures/"
make_ensemble_plot(combined_data, method_mapping, "regression_late", save_dir, col_names)



#-------------------------------#
#------ Cohort Comparison ------# 
#-------------------------------#

process_cohort <- function(results_dir, rho = 5) {
  full_path <- file.path(results_dir)
  
  # Determine file pattern based on cohort size
  file_pattern <- paste0("rho", as.character(rho), "_seed\\d+\\.txt$")
  file_list <- list.files(path = full_path, pattern = file_pattern, full.names = TRUE)
  data_list <- lapply(file_list, read_csv, col_names = TRUE, show_col_types = FALSE)
  all_data <- bind_rows(data_list)
  
  processed_data <- all_data %>%
    # filter(grepl("single", Method))%>%
    # mutate(Test_metric = as.numeric(Test_metric)) %>%
    # filter(grepl("^cohort", Method))%>%
    # filter(grepl("1", best_rho))%>%
    filter(grepl("cohort", Method))%>%
    fill(cohort_pairs)%>%
    mutate(
      CohortSize = cohort_size,
      Method = factor(Method, levels = c("cohort", "indep_cohort", "adversarial_cohort"))
    ) %>%
    mutate(
      # Convert 'cohort_pairs' to list of tuples
      cohort_pairs = strsplit(gsub("\\[|\\]", "", cohort_pairs), "\\), \\("),
      cohort_pairs = map(cohort_pairs, ~ gsub("^\\(|\\)$", "", .x)),  # Remove extra parentheses
      # Convert 'Test_metric' to list of values
      Test_metric = strsplit(gsub("\\[|\\]", "", Test_metric), ", ")
    ) %>%
    # Combine 'Test_metric' and 'cohort_pairs' into a list of data frames
    mutate(
      combined = map2(Test_metric, cohort_pairs, ~ data.frame(Test_metric = as.numeric(.x), cohort_pairs = .y))
    ) %>%
    # Drop the original 'Test_metric' and 'cohort_pairs' columns
    select(-Test_metric, -cohort_pairs) %>%
    # Unnest the combined list into separate rows
    unnest(combined) %>%
    # Split cohort_pairs into 'd1' and 'd2'
    separate(cohort_pairs, into = c("r1", "r2"), sep = ",", convert = TRUE) %>%
    group_by(Method, r1, r2) %>%
    summarise(
      mean_mse = mean(Test_metric, na.rm = TRUE),
      se_mse = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
      .groups = 'drop'
    )
  
  return(processed_data)
}

plot_error_heatmap <- function(data, metric = "mean_mse") {
  # List of methods to plot
  methods <- c("cohort", "indep_cohort", "adversarial_cohort")
  
  # Create a list to store individual plots
  plot_list <- list()
  
  # Find the global min and max for consistent color scaling across all plots
  global_min <- min(data[[metric]], na.rm = TRUE)
  global_max <- max(data[[metric]], na.rm = TRUE)
  
  for (method in methods) {
    # Filter data for the current method
    method_data <- data %>% filter(Method == method)
    
    # Create the heatmap for the current method
    p <- ggplot(method_data, aes(x = factor(r1), y = factor(r2), fill = !!sym(metric))) +
      geom_tile() +
      geom_text(aes(label = sprintf("%.2f", !!sym(metric))), 
                color = "white", size = 3) +
      scale_fill_viridis_c(option = "plasma", 
                           name = "Mean MSE", 
                           limits = c(global_min, global_max)) +
      labs(
        title = paste(method, "Performance"),
        x = "Modality 1 reduced dimension (r1)",
        y = "Modality 2 reduced dimension (r2)"
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, size = 14),
        legend.position = "right"
      )
    
    plot_list[[method]] <- p
  }
  
  # Arrange the plots in a grid
  combined_plot <- grid.arrange(grobs = plot_list, ncol = 3)
  
  return(combined_plot)
}

#results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_synthetic/results/"
results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_real/results/Neuron/negative_learning/"

save_dir <- "C:/Users/liang/Documents/GitHub/fusion/results/tables/"

cohort_data <- process_cohort(results_dir, rho=5)
error_heatmap <- plot_error_heatmap(cohort_data)


full_path <- file.path(results_dir)

# Determine file pattern based on cohort size
file_pattern <- paste0("rho", as.character(rho), "_seed\\d+\\.txt$")
file_list <- list.files(path = full_path, pattern = file_pattern, full.names = TRUE)
data_list <- lapply(file_list, read_csv, col_names = TRUE, show_col_types = FALSE)
all_data <- bind_rows(data_list)

c1 <- all_data %>%
  # filter(grepl("single", Method))%>%
  # mutate(Test_metric = as.numeric(Test_metric)) %>%
  filter(grepl("^cohort", Method))%>%
  filter(grepl("1", best_rho))%>%
  #filter(grepl("cohort", Method))%>%
  fill(cohort_pairs)%>%
  # mutate(
  #   CohortSize = cohort_size,
  #   Method = factor(Method, levels = c("cohort", "indep_cohort", "adversarial_cohort"))
  # ) %>%
  mutate(
    # Convert 'cohort_pairs' to list of tuples
    cohort_pairs = strsplit(gsub("\\[|\\]", "", cohort_pairs), "\\), \\("),
    cohort_pairs = map(cohort_pairs, ~ gsub("^\\(|\\)$", "", .x)),  # Remove extra parentheses
    # Convert 'Test_metric' to list of values
    Test_metric = strsplit(gsub("\\[|\\]", "", Test_metric), ", ")
  ) %>%
  # Combine 'Test_metric' and 'cohort_pairs' into a list of data frames
  mutate(
    combined = map2(Test_metric, cohort_pairs, ~ data.frame(Test_metric = as.numeric(.x), cohort_pairs = .y))
  ) %>%
  # Drop the original 'Test_metric' and 'cohort_pairs' columns
  select(-Test_metric, -cohort_pairs) %>%
  # Unnest the combined list into separate rows
  unnest(combined) %>%
  # Split cohort_pairs into 'd1' and 'd2'
  separate(cohort_pairs, into = c("r1", "r2"), sep = ",", convert = TRUE) %>%
  group_by(Method, r1, r2) %>%
  summarise(
    mean_mse = mean(Test_metric, na.rm = TRUE),
    se_mse = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )
c3 <- all_data %>%
  # filter(grepl("single", Method))%>%
  # mutate(Test_metric = as.numeric(Test_metric)) %>%
  filter(grepl("^cohort", Method))%>%
  filter(grepl("3", best_rho))%>%
  #filter(grepl("cohort", Method))%>%
  fill(cohort_pairs)%>%
  # mutate(
  #   CohortSize = cohort_size,
  #   Method = factor(Method, levels = c("cohort", "indep_cohort", "adversarial_cohort"))
  # ) %>%
  mutate(
    # Convert 'cohort_pairs' to list of tuples
    cohort_pairs = strsplit(gsub("\\[|\\]", "", cohort_pairs), "\\), \\("),
    cohort_pairs = map(cohort_pairs, ~ gsub("^\\(|\\)$", "", .x)),  # Remove extra parentheses
    # Convert 'Test_metric' to list of values
    Test_metric = strsplit(gsub("\\[|\\]", "", Test_metric), ", ")
  ) %>%
  # Combine 'Test_metric' and 'cohort_pairs' into a list of data frames
  mutate(
    combined = map2(Test_metric, cohort_pairs, ~ data.frame(Test_metric = as.numeric(.x), cohort_pairs = .y))
  ) %>%
  # Drop the original 'Test_metric' and 'cohort_pairs' columns
  select(-Test_metric, -cohort_pairs) %>%
  # Unnest the combined list into separate rows
  unnest(combined) %>%
  # Split cohort_pairs into 'd1' and 'd2'
  separate(cohort_pairs, into = c("r1", "r2"), sep = ",", convert = TRUE) %>%
  group_by(Method, r1, r2) %>%
  summarise(
    mean_mse = mean(Test_metric, na.rm = TRUE),
    se_mse = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )
c5 <- all_data %>%
  # filter(grepl("single", Method))%>%
  # mutate(Test_metric = as.numeric(Test_metric)) %>%
  filter(grepl("^cohort", Method))%>%
  filter(grepl("5", best_rho))%>%
  #filter(grepl("cohort", Method))%>%
  fill(cohort_pairs)%>%
  # mutate(
  #   CohortSize = cohort_size,
  #   Method = factor(Method, levels = c("cohort", "indep_cohort", "adversarial_cohort"))
  # ) %>%
  mutate(
    # Convert 'cohort_pairs' to list of tuples
    cohort_pairs = strsplit(gsub("\\[|\\]", "", cohort_pairs), "\\), \\("),
    cohort_pairs = map(cohort_pairs, ~ gsub("^\\(|\\)$", "", .x)),  # Remove extra parentheses
    # Convert 'Test_metric' to list of values
    Test_metric = strsplit(gsub("\\[|\\]", "", Test_metric), ", ")
  ) %>%
  # Combine 'Test_metric' and 'cohort_pairs' into a list of data frames
  mutate(
    combined = map2(Test_metric, cohort_pairs, ~ data.frame(Test_metric = as.numeric(.x), cohort_pairs = .y))
  ) %>%
  # Drop the original 'Test_metric' and 'cohort_pairs' columns
  select(-Test_metric, -cohort_pairs) %>%
  # Unnest the combined list into separate rows
  unnest(combined) %>%
  # Split cohort_pairs into 'd1' and 'd2'
  separate(cohort_pairs, into = c("r1", "r2"), sep = ",", convert = TRUE) %>%
  group_by(Method, r1, r2) %>%
  summarise(
    mean_mse = mean(Test_metric, na.rm = TRUE),
    se_mse = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )

c1 <- c1 %>%
  mutate(Method = if_else(Method == "cohort", "cohort1", Method))

c3 <- c3 %>%
  mutate(Method = if_else(Method == "cohort", "cohort3", Method))

c5 <- c5 %>%
  mutate(Method = if_else(Method == "cohort", "cohort5", Method))

cohort <- bind_rows(c1,c3,c5)

metric<-"mean_mse"
data<-cohort
methods <- c("cohort1", "cohort3", "cohort5")

# Create a list to store individual plots
plot_list <- list()

# Find the global min and max for consistent color scaling across all plots
global_min <- min(data[[metric]], na.rm = TRUE)
global_max <- max(data[[metric]], na.rm = TRUE)

for (method in methods) {
  # Filter data for the current method
  method_data <- data %>% filter(Method == method)
  
  # Create the heatmap for the current method
  p <- ggplot(method_data, aes(x = factor(r1), y = factor(r2), fill = !!sym(metric))) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", !!sym(metric))), 
              color = "white", size = 3) +
    scale_fill_viridis_c(option = "plasma", 
                         name = "Mean MSE", 
                         limits = c(global_min, global_max)) +
    labs(
      title = paste(method, "Performance"),
      x = "Modality 1 reduced dimension (r1)",
      y = "Modality 2 reduced dimension (r2)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(hjust = 0.5, size = 14),
      legend.position = "right"
    )
  
  plot_list[[method]] <- p
}

# Arrange the plots in a grid
combined_plot <- grid.arrange(grobs = plot_list, ncol = 3)

process_cohort <- function(results_dir) {
  full_path <- file.path(results_dir)
  
  # Determine file pattern based on cohort size
  file_pattern <- "seed\\d+\\.txt$"
  file_list <- list.files(path = full_path, pattern = file_pattern, full.names = TRUE)
  data_list <- lapply(file_list, read_csv, col_names = TRUE, show_col_types = FALSE)
  all_data <- bind_rows(data_list)
  
  processed_data <- all_data %>%
    # Group by random_state to ensure we're looking at each trial
    group_by(random_state) %>%
    # Filter to keep groups where cohort's best_rho is not 3
    filter(!(any(Method == "cohort" & best_rho == 0.99))) %>%
    # Ungroup to remove the grouping
    ungroup()%>%
    filter(grepl("cohort", Method))%>%
    fill(cohort_pairs)%>%
    mutate(
      CohortSize = cohort_size,
      Method = factor(Method, levels = c("cohort", "indep_cohort"))
    ) %>%
    mutate(
      # Convert 'cohort_pairs' to list of tuples
      cohort_pairs = strsplit(gsub("\\[|\\]", "", cohort_pairs), "\\), \\("),
      cohort_pairs = map(cohort_pairs, ~ gsub("^\\(|\\)$", "", .x)),  # Remove extra parentheses
      # Convert 'Test_metric' to list of values
      Test_metric = strsplit(gsub("\\[|\\]", "", Test_metric), ", ")
    ) %>%
    # Combine 'Test_metric' and 'cohort_pairs' into a list of data frames
    mutate(
      combined = map2(Test_metric, cohort_pairs, ~ data.frame(Test_metric = as.numeric(.x), cohort_pairs = .y))
    ) %>%
    # Drop the original 'Test_metric' and 'cohort_pairs' columns
    select(-Test_metric, -cohort_pairs) %>%
    # Unnest the combined list into separate rows
    unnest(combined) %>%
    # Split cohort_pairs into 'd1' and 'd2'
    separate(cohort_pairs, into = c("r1", "r2"), sep = ",", convert = TRUE) %>%
    group_by(Method, r1, r2) %>%
    summarise(
      mean_mse = mean(Test_metric, na.rm = TRUE),
      se_mse = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
      .groups = 'drop'
    )
  
  return(processed_data)
}

plot_error_heatmap <- function(data, metric = "mean_mse") {
  # List of methods to plot
  methods <- c("cohort", "indep_cohort")
  
  # Create a list to store individual plots
  plot_list <- list()
  
  # Find the global min and max for consistent color scaling across all plots
  global_min <- min(data[[metric]], na.rm = TRUE)
  global_max <- max(data[[metric]], na.rm = TRUE)
  
  for (method in methods) {
    # Filter data for the current method
    method_data <- data %>% filter(Method == method)
    
    # Create the heatmap for the current method
    p <- ggplot(method_data, aes(x = factor(r1), y = factor(r2), fill = !!sym(metric))) +
      geom_tile() +
      geom_text(aes(label = sprintf("%.2f", !!sym(metric))), 
                color = "white", size = 3) +
      scale_fill_viridis_c(option = "plasma", 
                           name = "Mean MSE", 
                           limits = c(global_min, global_max)) +
      labs(
        title = paste(method, "Performance"),
        x = "Modality 1 reduced dimension (r1)",
        y = "Modality 2 reduced dimension (r2)"
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, size = 14),
        legend.position = "right"
      )
    
    plot_list[[method]] <- p
  }
  
  # Arrange the plots in a grid
  combined_plot <- grid.arrange(grobs = plot_list, ncol = 3)
  
  return(combined_plot)
}

results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_real/results/Neuron/Barat/"
cohort_data <- process_cohort(results_dir)
error_heatmap <- plot_error_heatmap(cohort_data)




