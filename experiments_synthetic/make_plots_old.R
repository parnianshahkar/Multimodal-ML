# Load necessary libraries
library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(xtable)
library(stringr)
library(purrr)

#-------------------------------------------------------------------------------#
#----------------------------- Linear early ------------------------------------# 
#-------------------------------------------------------------------------------#
exp_name <- "regression_linear_early"
results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_synthetic/results/"
full_path <- paste(results_dir, exp_name, sep = "")

method_mapping <- c(
  "mod1" = "Mod 1",
  "mod2" = "Mod 2",
  "early_fusion" = "Early Fusion",
  "late_fusion" = "Late Fusion",
  "coop" = "Coop",
  "greedy_ensemble" = "Meta Fusion",
  "weighted_average" = "Weighted Average",
  "meta_learner" = "Meta Learner",
  "indep_best_single" = "Best Single (Indep.)",
  "best_single" = "Best Single"
)

file_list <- list.files(path = full_path, full.names = TRUE)
data_list <- list()

for (file in file_list) {
  # Read the file
  df <- read_csv(file, col_names = TRUE, show_col_types = FALSE)
  # Append the dataframe to the list
  data_list[[file]] <- df
}
all_data <- bind_rows(data_list)

# Non-cohort methods
non_cohort_data <- all_data %>%
  filter(!grepl("cohort", Method)) %>%
  mutate(Test_metric = as.numeric(Test_metric))

non_cohort_summary <- non_cohort_data %>%
  group_by(Method) %>%
  summarise(
    mean_metric = mean(Test_metric, na.rm = TRUE),
    se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),  # Calculate SE
    count = n(),  # Count the number of repetitions
    .groups = "drop"
  ) %>%
  rename(
    "Mean MSE" = mean_metric,
    "SE MSE" = se_metric
  )

print(non_cohort_summary, n=50)


#-----------------------------#
#---- Benchmark analysis -----# 
#-----------------------------#
# Plot the comparison with benchmarks
method_order <- c("mod1", "mod2", "early_fusion", "late_fusion",
                  "coop", 
                  "greedy_ensemble")

filtered_summary <- non_cohort_summary %>%
  filter(Method %in% method_order)%>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  mutate(Method = recode(Method, !!!method_mapping))%>%
  arrange(Method) %>%
  select(-count)

best_mse <- min(filtered_summary$`Mean MSE`)
filtered_summary <- filtered_summary %>%
  mutate(
    Method = as.character(Method),
    within_1se = (`Mean MSE` <= best_mse + `SE MSE`),  # Flag methods within 1 SE
    Method = ifelse(within_1se, paste0("\\textcolor{blue}{", Method, "}"), Method),
    `Mean MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`Mean MSE`, 2), "}"), round(`Mean MSE`, 2)),
    `SE MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`SE MSE`, 2), "}"), round(`SE MSE`, 2))
  ) %>%
  select(-within_1se)  # Remove the temporary flag column

latex_table <- xtable(filtered_summary)
print(latex_table, include.rownames = FALSE, type = "latex", sanitize.text.function = identity)


#-------------------------------------------------------------------------------
# Plot the comparison with different ensemble techniques
method_order <- c("indep_best_single", "best_single", "weighted_average", "meta_learner",
                  "greedy_ensemble")

# Filter for scale = 0 and reorder by specified method order
filtered_summary <- non_cohort_summary %>%
  filter(Method %in% method_order)%>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  mutate(Method = recode(Method, !!!method_mapping))%>%
  arrange(Method) %>%
  select(-count)

# Find the best performing model based on Mean MSE
best_mse <- min(filtered_summary$`Mean MSE`)

# Add bold formatting for methods within 1 SE of the best model
filtered_summary <- filtered_summary %>%
  mutate(
    Method = as.character(Method),
    within_1se = (`Mean MSE` <= best_mse + `SE MSE`),  # Flag methods within 1 SE
    Method = ifelse(within_1se, paste0("\\textcolor{blue}{", Method, "}"), Method),
    `Mean MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`Mean MSE`, 2), "}"), round(`Mean MSE`, 2)),
    `SE MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`SE MSE`, 2), "}"), round(`SE MSE`, 2))
  ) %>%
  select(-within_1se)  # Remove the temporary flag column


# Generate LaTeX table
latex_table <- xtable(filtered_summary)

print(latex_table, include.rownames = FALSE, type = "latex", sanitize.text.function = identity)

#--------------------------#
#---- Cohort analysis -----# 
#--------------------------#
cohort_data <- all_data %>%
  filter(grepl("cohort", Method))%>%
  fill(cohort_pairs)

# Check the best rhos 
best_rhos <- cohort_data %>%
  filter(Method == "cohort") %>%
  ggplot(aes(x = factor(best_rho))) +
  geom_bar(fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Column", x = "Column Name", y = "Count") +
  theme_minimal()
print(best_rhos)


# Unnest the list of test metrics and cohort dimensions into separate rows
cohort_data_parsed <- cohort_data %>%
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

# Plot the error heatmap with facet grid for method and scale
ggplot(cohort_data_parsed, aes(x = factor(r1), y = factor(r2), fill = mean_mse)) +
  geom_tile() +
  geom_text(aes(label = round(mean_mse, 3)), color = "white", size = 3) +  # Display up to 3 decimals
  facet_wrap( ~ Method) +  # Facet by 'scale' as rows and 'Method' as columns
  scale_fill_viridis_c(option = "plasma", name = "Mean MSE") +
  scale_x_discrete(limits = factor(c(0, 200, 300, 400, 500))) +  # Treat x-axis values as discrete
  scale_y_discrete(limits = factor(c(0, 100, 200, 300, 400))) +  # Treat y-axis values as discrete
  labs(
    title = "Mean MSE Heatmap by Method and Scale",
    x = "Modality 1 reduced dimension (r1)",
    y = "Modality 2 reduced dimension (r2)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 10)  # Adjust size of facet labels
  )

indep_cohort_mse <- cohort_data_parsed %>%
  filter(Method=='indep_cohort') %>%
  select(mean_mse)



#-------------------------------------------------------------------------------#
#---------------------------Quadratic early ------------------------------------# 
#-------------------------------------------------------------------------------#
exp_name <- "regression_quadratic_early"
results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_synthetic/results/"
full_path <- paste(results_dir, exp_name, sep = "")

method_mapping <- c(
  "mod1" = "Mod 1",
  "mod2" = "Mod 2",
  "early_fusion" = "Early Fusion",
  "late_fusion" = "Late Fusion",
  "coop" = "Coop",
  "greedy_ensemble" = "Meta Fusion",
  "weighted_average" = "Weighted Average",
  "meta_learner" = "Meta Learner",
  "indep_best_single" = "Best Single (Indep.)",
  "best_single" = "Best Single"
)

file_list <- list.files(path = full_path, full.names = TRUE)
data_list <- list()

for (file in file_list) {
  # Read the file
  df <- read_csv(file, col_names = TRUE, show_col_types = FALSE)
  # Append the dataframe to the list
  data_list[[file]] <- df
}
all_data <- bind_rows(data_list)

# Non-cohort methods
non_cohort_data <- all_data %>%
  filter(!grepl("cohort", Method)) %>%
  mutate(Test_metric = as.numeric(Test_metric))

non_cohort_summary <- non_cohort_data %>%
  group_by(Method) %>%
  summarise(
    mean_metric = mean(Test_metric, na.rm = TRUE),
    se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),  # Calculate SE
    count = n(),  # Count the number of repetitions
    .groups = "drop"
  ) %>%
  rename(
    "Mean MSE" = mean_metric,
    "SE MSE" = se_metric
  )

print(non_cohort_summary, n=50)


#-----------------------------#
#---- Benchmark analysis -----# 
#-----------------------------#
# Plot the comparison with benchmarks
method_order <- c("mod1", "mod2", "early_fusion", "late_fusion",
                  "coop", 
                  "greedy_ensemble")

filtered_summary <- non_cohort_summary %>%
  filter(Method %in% method_order)%>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  mutate(Method = recode(Method, !!!method_mapping))%>%
  arrange(Method) %>%
  select(-count)

best_mse <- min(filtered_summary$`Mean MSE`)
filtered_summary <- filtered_summary %>%
  mutate(
    Method = as.character(Method),
    within_1se = (`Mean MSE` <= best_mse + `SE MSE`),  # Flag methods within 1 SE
    Method = ifelse(within_1se, paste0("\\textcolor{blue}{", Method, "}"), Method),
    `Mean MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`Mean MSE`, 2), "}"), round(`Mean MSE`, 2)),
    `SE MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`SE MSE`, 2), "}"), round(`SE MSE`, 2))
  ) %>%
  select(-within_1se)  # Remove the temporary flag column

latex_table <- xtable(filtered_summary)
print(latex_table, include.rownames = FALSE, type = "latex", sanitize.text.function = identity)


#-------------------------------------------------------------------------------
# Plot the comparison with different ensemble techniques
method_order <- c("indep_best_single", "best_single", "weighted_average", "meta_learner",
                  "greedy_ensemble")

# Filter for scale = 0 and reorder by specified method order
filtered_summary <- non_cohort_summary %>%
  filter(Method %in% method_order)%>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  mutate(Method = recode(Method, !!!method_mapping))%>%
  arrange(Method) %>%
  select(-count)

# Find the best performing model based on Mean MSE
best_mse <- min(filtered_summary$`Mean MSE`)

# Add bold formatting for methods within 1 SE of the best model
filtered_summary <- filtered_summary %>%
  mutate(
    Method = as.character(Method),
    within_1se = (`Mean MSE` <= best_mse + `SE MSE`),  # Flag methods within 1 SE
    Method = ifelse(within_1se, paste0("\\textcolor{blue}{", Method, "}"), Method),
    `Mean MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`Mean MSE`, 2), "}"), round(`Mean MSE`, 2)),
    `SE MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`SE MSE`, 2), "}"), round(`SE MSE`, 2))
  ) %>%
  select(-within_1se)  # Remove the temporary flag column


# Generate LaTeX table
latex_table <- xtable(filtered_summary)

print(latex_table, include.rownames = FALSE, type = "latex", sanitize.text.function = identity)


#--------------------------#
#---- Cohort analysis -----# 
#--------------------------#
cohort_data <- all_data %>%
  filter(grepl("cohort", Method))%>%
  filter(d1 == 2000, scale %in% c(0, 0.5, 1, 5)) %>%
  fill(cohort_pairs) %>%
  select(-best_rho)


# Unnest the list of test metrics and cohort dimensions into separate rows
cohort_data_parsed <- cohort_data %>%
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
    group_by(Method, r1, r2, scale) %>%
    summarise(
      mean_mse = mean(Test_metric, na.rm = TRUE),
      se_mse = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
      .groups = 'drop'
    )

# Plot the error heatmap with facet grid for method and scale
ggplot(cohort_data_parsed, aes(x = r1, y = r2, fill = mean_mse)) +
  geom_tile() +
  geom_text(aes(label = round(mean_mse, 3)), color = "white", size = 3) +  # Display up to 3 decimals
  facet_grid(scale ~ Method) +  # Facet by 'scale' as rows and 'Method' as columns
  scale_fill_viridis_c(option = "plasma", name = "Mean MSE") +
  labs(
    title = "Mean MSE Heatmap by Method and Scale",
    x = "Modality 1 reduced dimension (r1)",
    y = "Modality 2 reduced dimension (r2)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 10)  # Adjust size of facet labels
  )

indep_cohort_mse <- cohort_data_parsed %>%
                      filter(Method=='indep_cohort') %>%
                      select(mean_mse)



#-------------------------------------------------------------------------------#
#----------------------------- Linear Late  ------------------------------------# 
#-------------------------------------------------------------------------------#
exp_name <- "regression_linear_late"
results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_synthetic/results/"
full_path <- paste0(results_dir, exp_name)

method_mapping <- c(
  "mod1" = "Mod 1",
  "mod2" = "Mod 2",
  "early_fusion" = "Early Fusion",
  "late_fusion" = "Late Fusion",
  "coop" = "Coop",
  "greedy_ensemble" = "Meta Fusion",
  "weighted_average" = "Weighted Average",
  "meta_learner" = "Meta Learner",
  "indep_best_single" = "Best Single (Indep.)",
  "best_single" = "Best Single"
)

file_list <- list.files(path = full_path, full.names = TRUE)
data_list <- list()

for (file in file_list) {
  # Read the file
  df <- read_csv(file, col_names = TRUE, show_col_types = FALSE)
  # Append the dataframe to the list
  data_list[[file]] <- df
}
all_data <- bind_rows(data_list)

# Non-cohort methods
non_cohort_data <- all_data %>%
  filter(!grepl("cohort", Method)) %>%
  mutate(Test_metric = as.numeric(Test_metric))

non_cohort_summary <- non_cohort_data %>%
  group_by(Method) %>%
  summarise(
    mean_metric = mean(Test_metric, na.rm = TRUE),
    se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),  # Calculate SE
    count = n(),  # Count the number of repetitions
    .groups = "drop"
  ) %>%
  rename(
    "Mean MSE" = mean_metric,
    "SE MSE" = se_metric
  )

print(non_cohort_summary, n=50)


#-----------------------------#
#---- Benchmark analysis -----# 
#-----------------------------#
# Plot the comparison with benchmarks
method_order <- c("mod1", "mod2", "early_fusion", "late_fusion",
                  "coop", 
                  "greedy_ensemble")

filtered_summary <- non_cohort_summary %>%
  filter(Method %in% method_order)%>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  mutate(Method = recode(Method, !!!method_mapping))%>%
  arrange(Method) %>%
  select(-count)

best_mse <- min(filtered_summary$`Mean MSE`)
filtered_summary <- filtered_summary %>%
  mutate(
    Method = as.character(Method),
    within_1se = (`Mean MSE` <= best_mse + `SE MSE`),  # Flag methods within 1 SE
    Method = ifelse(within_1se, paste0("\\textcolor{blue}{", Method, "}"), Method),
    `Mean MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`Mean MSE`, 2), "}"), round(`Mean MSE`, 2)),
    `SE MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`SE MSE`, 2), "}"), round(`SE MSE`, 2))
  ) %>%
  select(-within_1se)  # Remove the temporary flag column

latex_table <- xtable(filtered_summary)
print(latex_table, include.rownames = FALSE, type = "latex", sanitize.text.function = identity)


#-------------------------------------------------------------------------------
# Plot the comparison with different ensemble techniques
method_order <- c("indep_best_single", "best_single", "weighted_average", "meta_learner",
                  "greedy_ensemble")

# Filter for scale = 0 and reorder by specified method order
filtered_summary <- non_cohort_summary %>%
  filter(Method %in% method_order)%>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  mutate(Method = recode(Method, !!!method_mapping))%>%
  arrange(Method) %>%
  select(-count)

# Find the best performing model based on Mean MSE
best_mse <- min(filtered_summary$`Mean MSE`)

# Add bold formatting for methods within 1 SE of the best model
filtered_summary <- filtered_summary %>%
  mutate(
    Method = as.character(Method),
    within_1se = (`Mean MSE` <= best_mse + `SE MSE`),  # Flag methods within 1 SE
    Method = ifelse(within_1se, paste0("\\textcolor{blue}{", Method, "}"), Method),
    `Mean MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`Mean MSE`, 2), "}"), round(`Mean MSE`, 2)),
    `SE MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`SE MSE`, 2), "}"), round(`SE MSE`, 2))
  ) %>%
  select(-within_1se)  # Remove the temporary flag column


# Generate LaTeX table
latex_table <- xtable(filtered_summary)

print(latex_table, include.rownames = FALSE, type = "latex", sanitize.text.function = identity)

#--------------------------#
#---- Cohort analysis -----# 
#--------------------------#
cohort_data <- all_data %>%
  filter(grepl("cohort", Method))%>%
  filter(extractor=='separate')%>%
  fill(cohort_pairs) %>%
  select(-best_rho)


# Unnest the list of test metrics and cohort dimensions into separate rows
cohort_data_parsed <- cohort_data %>%
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
  group_by(Method, r1, r2, scale) %>%
  summarise(
    mean_mse = mean(Test_metric, na.rm = TRUE),
    se_mse = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )

# Plot the error heatmap with facet grid for method and scale
ggplot(cohort_data_parsed, aes(x = r1, y = r2, fill = mean_mse)) +
  geom_tile() +
  geom_text(aes(label = round(mean_mse, 3)), color = "white", size = 3) +  # Display up to 3 decimals
  facet_grid(scale ~ Method) +  # Facet by 'scale' as rows and 'Method' as columns
  scale_fill_viridis_c(option = "plasma", name = "Mean MSE") +
  labs(
    title = "Mean MSE Heatmap by Method and Scale",
    x = "Modality 1 reduced dimension (r1)",
    y = "Modality 2 reduced dimension (r2)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 10)  # Adjust size of facet labels
  )

indep_cohort_mse <- cohort_data_parsed %>%
  filter(Method=='indep_cohort') %>%
  select(mean_mse)



#-------------------------------------------------------------------------------#
#---------------------------Quadratic Late- ------------------------------------# 
#-------------------------------------------------------------------------------#
exp_name <- "regression_quadratic_late"
results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_synthetic/results/"
full_path <- paste0(results_dir, exp_name)

method_mapping <- c(
  "mod1" = "Mod 1",
  "mod2" = "Mod 2",
  "early_fusion" = "Early Fusion",
  "late_fusion" = "Late Fusion",
  "coop" = "Coop",
  "greedy_ensemble" = "Meta Fusion",
  "weighted_average" = "Weighted Average",
  "meta_learner" = "Meta Learner",
  "indep_best_single" = "Best Single (Indep.)",
  "best_single" = "Best Single"
)

file_list <- list.files(path = full_path, full.names = TRUE)
data_list <- list()

for (file in file_list) {
  # Read the file
  df <- read_csv(file, col_names = TRUE, show_col_types = FALSE)
  # Append the dataframe to the list
  data_list[[file]] <- df
}
all_data <- bind_rows(data_list)

# Non-cohort methods
non_cohort_data <- all_data %>%
  filter(extractor=='PCA') %>%
  filter(!grepl("cohort", Method)) %>%
  mutate(Test_metric = as.numeric(Test_metric))

non_cohort_summary <- non_cohort_data %>%
  group_by(Method, extractor) %>%
  summarise(
    mean_metric = mean(Test_metric, na.rm = TRUE),
    se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),  # Calculate SE
    count = n(),  # Count the number of repetitions
    .groups = "drop"
  ) %>%
  rename(
    "Mean MSE" = mean_metric,
    "SE MSE" = se_metric
  )

print(non_cohort_summary, n=50)


#-----------------------------#
#---- Benchmark analysis -----# 
#-----------------------------#
# Plot the comparison with benchmarks
method_order <- c("mod1", "mod2", "early_fusion", "late_fusion",
                  "coop", 
                  "greedy_ensemble")

filtered_summary <- non_cohort_summary %>%
  filter(Method %in% method_order)%>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  mutate(Method = recode(Method, !!!method_mapping))%>%
  arrange(Method) %>%
  select(-count, -extractor)

best_mse <- min(filtered_summary$`Mean MSE`)
filtered_summary <- filtered_summary %>%
  mutate(
    Method = as.character(Method),
    within_1se = (`Mean MSE` <= best_mse + `SE MSE`),  # Flag methods within 1 SE
    Method = ifelse(within_1se, paste0("\\textcolor{blue}{", Method, "}"), Method),
    `Mean MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`Mean MSE`, 2), "}"), round(`Mean MSE`, 2)),
    `SE MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`SE MSE`, 2), "}"), round(`SE MSE`, 2))
  ) %>%
  select(-within_1se)  # Remove the temporary flag column

latex_table <- xtable(filtered_summary)
print(latex_table, include.rownames = FALSE, type = "latex", sanitize.text.function = identity)


#-------------------------------------------------------------------------------
# Plot the comparison with different ensemble techniques
method_order <- c("indep_best_single", "best_single", "weighted_average", "meta_learner",
                  "greedy_ensemble")

# Filter for scale = 0 and reorder by specified method order
filtered_summary <- non_cohort_summary %>%
  filter(Method %in% method_order)%>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  mutate(Method = recode(Method, !!!method_mapping))%>%
  arrange(Method) %>%
  select(-count, -extractor)

# Find the best performing model based on Mean MSE
best_mse <- min(filtered_summary$`Mean MSE`)

# Add bold formatting for methods within 1 SE of the best model
filtered_summary <- filtered_summary %>%
  mutate(
    Method = as.character(Method),
    within_1se = (`Mean MSE` <= best_mse + `SE MSE`),  # Flag methods within 1 SE
    Method = ifelse(within_1se, paste0("\\textcolor{blue}{", Method, "}"), Method),
    `Mean MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`Mean MSE`, 2), "}"), round(`Mean MSE`, 2)),
    `SE MSE` = ifelse(within_1se, paste0("\\textcolor{blue}{", round(`SE MSE`, 2), "}"), round(`SE MSE`, 2))
  ) %>%
  select(-within_1se)  # Remove the temporary flag column


# Generate LaTeX table
latex_table <- xtable(filtered_summary)

print(latex_table, include.rownames = FALSE, type = "latex", sanitize.text.function = identity)

#--------------------------#
#---- Cohort analysis -----# 
#--------------------------#
# Check the best rhos 
best_rhos <- non_cohort_data %>%
  filter(Method == "meta_learner") %>%
  ggplot(aes(x = factor(best_rho))) +
  geom_bar(fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Column", x = "Column Name", y = "Count") +
  theme_minimal()
print(best_rhos)


cohort_data <- all_data %>%
  filter(grepl("cohort", Method))%>%
  filter(extractor=='PCA')%>%
  fill(cohort_pairs) %>%
  select(-best_rho)


# Unnest the list of test metrics and cohort dimensions into separate rows
cohort_data_parsed <- cohort_data %>%
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

# Plot the error heatmap with facet grid for method and scale
ggplot(cohort_data_parsed, aes(x = factor(r1), y = factor(r2), fill = mean_mse)) +
  geom_tile() +
  geom_text(aes(label = round(mean_mse, 3)), color = "white", size = 3) +  # Display up to 3 decimals
  facet_grid(~ Method) +  # Facet by 'scale' as rows and 'Method' as columns
  scale_fill_viridis_c(option = "plasma", name = "Mean MSE") +
  scale_x_discrete(limits = factor(c(0, 60, 80, 100, 120))) +  # Treat x-axis values as discrete
  scale_y_discrete(limits = factor(c(0, 60, 80, 100, 120))) +  # Treat y-axis values as discrete
  labs(
    title = "Mean MSE Heatmap by Method and Scale",
    x = "Modality 1 reduced dimension (r1)",
    y = "Modality 2 reduced dimension (r2)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 10)  # Adjust size of facet labels
  )

indep_cohort_mse <- cohort_data_parsed %>%
  filter(Method=='indep_cohort') %>%
  select(mean_mse)



#------------------------------------------------------------------------------
exp_name <- "regression_quadratic_late"
results_dir <- "C:/Users/liang/Documents/GitHub/fusion/experiments_large_scale/results/"
full_path <- paste0(results_dir, exp_name)

# Define the file pattern (customizable for each experiment)
file_list <- list.files(path = full_path, full.names = TRUE)

# Define the desired order of methods
method_order <- c("mod1", "mod2", "early_fusion", "late_fusion",
                  "coop",
                  "best_single", "indep_best_single", 
                  "meta_learner", "greedy_ensemble", 
                  "weighted_average")

# Initialize an empty list to store data from all files
data_list <- list()

# Loop through all files and load them
for (file in file_list) {
  # Read the file
  df <- read_csv(file, col_names = TRUE, show_col_types = FALSE)
  # Append the dataframe to the list
  data_list[[file]] <- df
}

# Combine all the data into one dataframe
all_data <- bind_rows(data_list)


# Non-cohort methods
non_cohort_data <- all_data %>%
  filter(extractor=='PCA', scale==5) %>%
  # filter(!grepl("encoder", Method)) %>%
  mutate(Test_metric = as.numeric(Test_metric))

non_cohort_summary <- non_cohort_data %>%
  group_by(scale, Method, extractor) %>%
  summarise(
    mean_metric = mean(Test_metric, na.rm = TRUE),
    se_metric = sd(Test_metric, na.rm = TRUE) / sqrt(n()),  # Calculate SE
    count = n(),  # Count the number of repetitions
    .groups = "drop"
  ) %>%
  rename(
    "Mean MSE" = mean_metric,
    "SE MSE" = se_metric
  )

# Print non-cohort summary
print(non_cohort_summary, n=50)


# Filter for scale = 0 and reorder by specified method order
filtered_summary <- non_cohort_summary %>%
  filter(Method %in% method_order) %>%
  mutate(Method = factor(Method, levels = method_order)) %>%
  arrange(Method) %>%
  select(-count, -extractor,-scale)%>% 
  mutate(
    Method = gsub("_", " ", Method),                  # Replace underscores with spaces
    Method = str_replace(Method, "indep", "indep."),  # Replace 'indep' with 'indep.'
    Method = str_to_sentence(Method)                  # Capitalize the first letter
  )

# Find the best performing model based on Mean MSE
best_mse <- min(filtered_summary$`Mean MSE`)

# Add bold formatting for methods within 1 SE of the best model
filtered_summary <- filtered_summary %>%
  mutate(
    within_1se = (`Mean MSE` <= best_mse + `SE MSE`),  # Flag methods within 1 SE
    Method = ifelse(within_1se, paste0("\\textbf{", Method, "}"), Method),
    `Mean MSE` = ifelse(within_1se, paste0("\\textbf{", round(`Mean MSE`, 2), "}"), round(`Mean MSE`, 2)),
    `SE MSE` = ifelse(within_1se, paste0("\\textbf{", round(`SE MSE`, 2), "}"), round(`SE MSE`, 2))
  ) %>%
  select(-within_1se)  # Remove the temporary flag column


# Generate LaTeX table
latex_table <- xtable(filtered_summary)

print(latex_table, include.rownames = FALSE, type = "latex", sanitize.text.function = identity)


#--------------------------#
#---- Cohort analysis -----# 
#--------------------------#
# Check the best rhos 
best_rhos <- non_cohort_data %>%
  filter(Method == "meta_learner") %>%
  ggplot(aes(x = factor(best_rho))) +
  geom_bar(fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Column", x = "Column Name", y = "Count") +
  theme_minimal()
print(best_rhos)


cohort_data <- all_data %>%
  filter(grepl("cohort", Method))%>%
  filter(extractor=='PCA', scale==5)%>%
  fill(cohort_pairs) %>%
  select(-best_rho)


# Unnest the list of test metrics and cohort dimensions into separate rows
cohort_data_parsed <- cohort_data %>%
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
  group_by(Method, r1, r2, scale) %>%
  summarise(
    mean_mse = mean(Test_metric, na.rm = TRUE),
    se_mse = sd(Test_metric, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )

# # Plot the error heatmap with facet grid for method and scale
# ggplot(cohort_data_parsed, aes(x = r1, y = r2, fill = mean_mse)) +
#   geom_tile() +
#   geom_text(aes(label = round(mean_mse, 3)), color = "white", size = 3) +  # Display up to 3 decimals
#   facet_grid(scale ~ Method) +  # Facet by 'scale' as rows and 'Method' as columns
#   scale_fill_viridis_c(option = "plasma", name = "Mean MSE") +
#   labs(
#     title = "Mean MSE Heatmap by Method and Scale",
#     x = "Modality 1 reduced dimension (r1)",
#     y = "Modality 2 reduced dimension (r2)"
#   ) +
#   theme_minimal() +
#   theme(
#     axis.text.x = element_text(angle = 45, hjust = 1),
#     strip.text = element_text(size = 10)  # Adjust size of facet labels
#   )

# Plot the error heatmap with facet grid for method and scale
ggplot(cohort_data_parsed, aes(x = factor(r1), y = factor(r2), fill = mean_mse)) +
  geom_tile() +
  geom_text(aes(label = round(mean_mse, 1)), color = "white", size = 3) +  # Display up to 3 decimals
  facet_grid(scale ~ Method) +  # Facet by 'scale' as rows and 'Method' as columns
  scale_fill_viridis_c(option = "plasma", name = "Mean MSE") +
  scale_x_discrete(limits = factor(c(0, 40, 50, 60, 70, 80, 90, 100, 110, 120))) +  # Treat x-axis values as discrete
  scale_y_discrete(limits = factor(c(0, 40, 50, 60, 70, 80, 90, 100, 110, 120))) +  # Treat y-axis values as discrete
  labs(
    title = "Mean MSE Heatmap by Method and Scale",
    x = "Modality 1 reduced dimension (r1)",
    y = "Modality 2 reduced dimension (r2)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 10)  # Adjust size of facet labels
  )

indep_cohort_mse <- cohort_data_parsed %>%
  filter(Method=='indep_cohort') %>%
  select(mean_mse)

