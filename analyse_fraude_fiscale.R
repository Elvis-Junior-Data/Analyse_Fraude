################################################################################
# PROJET INTÉGRATEUR — DÉTECTION DE FRAUDE FISCALE
# M1 R Avancée — Examen Final (02.06.2026)
# Adult Income Dataset (UCI)
################################################################################

# ============================================================
# PACKAGES
# ============================================================
library(tidyverse)
library(janitor)
library(tidymodels)
library(themis)        # SMOTE
library(vip)           # variable importance
library(probably)      # calibration / seuils
library(gt)            # tableaux
library(corrplot)      # heatmap corrélation
library(pROC)          # ROC
library(yardstick)     # métriques

set.seed(2026)


################################################################################
# 01 — IMPORT & NETTOYAGE
################################################################################

col_names <- c(
  "age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "sex",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
)

# Lecture train
adult_train_raw <- read_csv(
  "data/adult.data",
  col_names = col_names,
  trim_ws   = TRUE,
  na        = c("", "NA", "?"),
  show_col_types = FALSE
)

# Lecture test (première ligne = commentaire à ignorer)
adult_test_raw <- read_csv(
  "data/adult.test",
  col_names = col_names,
  skip      = 1,
  trim_ws   = TRUE,
  na        = c("", "NA", "?"),
  show_col_types = FALSE
)

# Fusion
adult_raw <- bind_rows(
  adult_train_raw %>% mutate(split = "train"),
  adult_test_raw  %>% mutate(split = "test")
)

# ---- Nettoyage ----
adult_clean <- adult_raw %>%
  # Nettoyer la variable cible (le fichier test ajoute un ".")
  mutate(income = str_remove(income, "\\.") %>% str_trim()) %>%
  # Standardiser les noms avec janitor
  clean_names() %>%
  # Convertir en facteurs
  mutate(
    across(
      c(workclass, education, marital_status, occupation,
        relationship, race, sex, native_country, income, split),
      as.factor
    ),
    # income : réordonner pour que >50K soit le niveau positif
    income = fct_relevel(income, ">50K")
  ) %>%
  # Supprimer les NA
  drop_na()

cat("Dimensions après nettoyage :", nrow(adult_clean), "×", ncol(adult_clean), "\n")
cat("Valeurs manquantes restantes :", sum(is.na(adult_clean)), "\n")
glimpse(adult_clean)


################################################################################
# 02 — ANALYSE EXPLORATOIRE (EDA)
################################################################################

# ---- Distribution de la variable cible ----
p1 <- adult_clean %>%
  count(income) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = income, y = n, fill = income)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)), vjust = -0.3, size = 4) +
  scale_fill_manual(values = c("#E74C3C", "#2ECC71")) +
  labs(
    title    = "Distribution de la variable cible `income`",
    subtitle = "Déséquilibre de classes notable",
    x = NULL, y = "Effectif"
  ) +
  theme_minimal(base_size = 13)

print(p1)

# ---- Distribution de l'âge par classe ----
p2 <- adult_clean %>%
  ggplot(aes(x = age, fill = income)) +
  geom_histogram(binwidth = 3, position = "dodge", alpha = 0.85) +
  scale_fill_manual(values = c("#E74C3C", "#3498DB")) +
  labs(title = "Distribution de l'âge par classe de revenu",
       x = "Âge", y = "Effectif", fill = "Revenu") +
  theme_minimal(base_size = 13)

print(p2)

# ---- Capital gain — boxplot ----
p3 <- adult_clean %>%
  filter(capital_gain > 0) %>%
  ggplot(aes(x = income, y = capital_gain, fill = income)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  scale_y_log10(labels = scales::dollar) +
  scale_fill_manual(values = c("#E74C3C", "#3498DB")) +
  labs(title = "Capital gain (non-nul) par classe de revenu — échelle log",
       x = NULL, y = "Capital gain (log$)", fill = "Revenu") +
  theme_minimal(base_size = 13)

print(p3)

# ---- Heures travaillées / semaine ----
p4 <- adult_clean %>%
  ggplot(aes(x = hours_per_week, fill = income)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#E74C3C", "#3498DB")) +
  labs(title = "Densité des heures travaillées par semaine",
       x = "Heures/semaine", y = "Densité", fill = "Revenu") +
  theme_minimal(base_size = 13)

print(p4)

# ---- Taux de >50K par niveau d'éducation ----
p5 <- adult_clean %>%
  group_by(education) %>%
  summarise(
    taux_risque = mean(income == ">50K"),
    n = n()
  ) %>%
  filter(n > 50) %>%
  mutate(education = fct_reorder(education, taux_risque)) %>%
  ggplot(aes(x = education, y = taux_risque, fill = taux_risque)) +
  geom_col(show.legend = FALSE) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_gradient(low = "#85C1E9", high = "#E74C3C") +
  coord_flip() +
  labs(title = "Taux de revenu >50K par niveau d'éducation",
       x = NULL, y = "Taux de fraude proxy") +
  theme_minimal(base_size = 13)

print(p5)

# ---- Matrice de corrélation (variables numériques) ----
vars_num <- adult_clean %>%
  select(age, fnlwgt, education_num, capital_gain, capital_loss,
         hours_per_week) %>%
  cor()

corrplot(vars_num,
         method = "color",
         type   = "upper",
         tl.col = "black",
         addCoef.col = "black",
         number.cex  = 0.75,
         title  = "Matrice de corrélation — variables numériques",
         mar    = c(0, 0, 1, 0))

# ---- Profil moyen par classe ----
adult_clean %>%
  group_by(income) %>%
  summarise(
    age_moyen            = mean(age),
    heures_moyen         = mean(hours_per_week),
    capital_gain_moyen   = mean(capital_gain),
    capital_loss_moyen   = mean(capital_loss),
    education_num_moyen  = mean(education_num)
  ) %>%
  gt() %>%
  tab_header(title = "Profil moyen par classe de revenu") %>%
  fmt_number(columns = where(is.numeric), decimals = 2)


################################################################################
# 03 — FEATURE ENGINEERING
################################################################################

adult_fe <- adult_clean %>%
  mutate(
    # Tranche d'âge
    age_group = case_when(
      age < 25            ~ "Jeune (<25)",
      age >= 25 & age < 40 ~ "Adulte (25-39)",
      age >= 40 & age < 55 ~ "Expérimenté (40-54)",
      age >= 55            ~ "Senior (55+)"
    ) %>% factor(levels = c("Jeune (<25)", "Adulte (25-39)",
                             "Expérimenté (40-54)", "Senior (55+)")),

    # Solde net du capital
    capital_net = capital_gain - capital_loss,

    # Indicateur hautes heures (> 50 h/sem)
    high_hours = factor(if_else(hours_per_week > 50, "Oui", "Non")),

    # Score d'éducation ordinal (identique à education_num mais renommé)
    education_score = education_num,

    # Indicateur d'activité en capital
    has_capital = factor(if_else(capital_gain > 0 | capital_loss > 0,
                                 "Oui", "Non"))
  )

# Vérification
cat("Variables créées :\n")
adult_fe %>%
  select(age_group, capital_net, high_hours, education_score, has_capital) %>%
  summary() %>%
  print()

# Distribution avant/après capital_net
p6 <- adult_fe %>%
  filter(abs(capital_net) > 0) %>%
  ggplot(aes(x = capital_net, fill = income)) +
  geom_histogram(bins = 40, position = "dodge", alpha = 0.8) +
  scale_x_continuous(labels = scales::dollar) +
  scale_fill_manual(values = c("#E74C3C", "#3498DB")) +
  labs(title = "Distribution de capital_net (non-nul) par classe",
       x = "Capital net ($)", y = "Effectif", fill = "Revenu") +
  theme_minimal(base_size = 13)

print(p6)


################################################################################
# 04 — MODÉLISATION (Tidymodels)
################################################################################

# Travailler uniquement sur le train split
data_model <- adult_fe %>% filter(split == "train") %>% select(-split)

# ---- Découpage train / validation ----
splits <- initial_split(data_model, prop = 0.8, strata = income)
train_data <- training(splits)
test_data  <- testing(splits)

# ---- Folds (5-fold stratifié) ----
folds <- vfold_cv(train_data, v = 5, strata = income)

# ---- Recipe ----
rec <- recipe(income ~ ., data = train_data) %>%
  # Supprimer variables redondantes / de pondération
  step_rm(fnlwgt, education) %>%
  # Encodage one-hot des catégorielles
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # Normalisation des numériques
  step_normalize(all_numeric_predictors()) %>%
  # Supprimer colonnes à variance nulle
  step_zv(all_predictors()) %>%
  # SMOTE pour rééquilibrage
  step_smote(income, over_ratio = 0.8)

# ---- Modèle : Régression logistique pénalisée (elastic net) ----
log_spec <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# ---- Workflow ----
log_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(log_spec)

# ---- Grille de recherche ----
log_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(range = c(0, 1)),
  levels = 5
)

# ---- Tuning ----
cat("Optimisation des hyperparamètres (5-fold CV)...\n")

log_res <- tune_grid(
  log_wf,
  resamples = folds,
  grid      = log_grid,
  metrics   = metric_set(roc_auc, f_meas),
  control   = control_grid(verbose = TRUE, save_pred = TRUE)
)

# ---- Meilleurs hyperparamètres ----
best_params_auc <- select_best(log_res, metric = "roc_auc")
best_params_f1  <- select_best(log_res, metric = "f_meas")

cat("\nMeilleurs paramètres (AUC-ROC) :\n")
print(best_params_auc)

cat("\nMeilleurs paramètres (F1) :\n")
print(best_params_f1)

# Visualisation des résultats de tuning
autoplot(log_res)


################################################################################
# 05 — ÉVALUATION DU MEILLEUR MODÈLE
################################################################################

# ---- Finalisation sur AUC ----
final_wf <- finalize_workflow(log_wf, best_params_auc)

# ---- Ajustement final sur train complet + prédiction sur test ----
final_fit <- last_fit(final_wf, splits)

# ---- Métriques ----
metriques <- collect_metrics(final_fit)
cat("\n=== Métriques finales sur le test set ===\n")
print(metriques)

# ---- Prédictions ----
preds <- collect_predictions(final_fit)

# Matrice de confusion
conf_mat_res <- conf_mat(preds, truth = income, estimate = .pred_class)
cat("\nMatrice de confusion :\n")
print(conf_mat_res)
autoplot(conf_mat_res, type = "heatmap") +
  labs(title = "Matrice de confusion — modèle final")

# Courbe ROC
roc_plot <- preds %>%
  roc_curve(truth = income, `.pred_>50K`) %>%
  autoplot() +
  labs(title = sprintf("Courbe ROC — AUC = %.3f",
                       roc_auc(preds, truth = income, `.pred_>50K`)$.estimate))
print(roc_plot)

# Courbe PR (Precision-Recall)
pr_plot <- preds %>%
  pr_curve(truth = income, `.pred_>50K`) %>%
  autoplot() +
  labs(title = "Courbe Precision-Recall")
print(pr_plot)

# F1, précision, rappel
preds %>%
  metrics(truth = income, estimate = .pred_class) %>%
  bind_rows(
    preds %>% f_meas(truth = income, estimate = .pred_class),
    preds %>% precision(truth = income, estimate = .pred_class),
    preds %>% recall(truth = income, estimate = .pred_class)
  ) %>%
  print()

# ---- Importance des variables ----
final_model <- extract_fit_parsnip(final_fit)

vip_plot <- vip(final_model,
                num_features = 20,
                geom = "col",
                aesthetics = list(fill = "#2980B9", alpha = 0.85)) +
  labs(title = "Top 20 — Importance des variables (modèle final)")
print(vip_plot)

# ---- Analyse des erreurs ----
preds %>%
  filter(.pred_class != income) %>%
  count(income, .pred_class) %>%
  mutate(type_erreur = if_else(income == ">50K", "Faux Négatif", "Faux Positif")) %>%
  print()

# ---- Sauvegarde du modèle final ----
modele_final <- extract_workflow(final_fit)
saveRDS(modele_final, file = "modele_fraude_final.rds")
cat("\nModèle sauvegardé : modele_fraude_final.rds\n")
