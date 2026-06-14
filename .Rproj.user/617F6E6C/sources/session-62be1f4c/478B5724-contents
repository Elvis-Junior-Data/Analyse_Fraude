################################################################################
# LIVRABLE 3 — APPLICATION SHINY : DÉTECTION DE FRAUDE FISCALE
# Fichier : R/app_fraude.R
# M1 R Avancée — Examen Final 2026
################################################################################

library(shiny)
library(shinydashboard)
library(tidyverse)
library(janitor)
library(tidymodels)
library(DT)
library(plotly)
library(scales)
library(yardstick)

# ============================================================
# CHARGEMENT DES DONNÉES & DU MODÈLE
# ============================================================

# --- Import & nettoyage ---
col_names <- c(
  "age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "sex",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
)

load_data <- function() {
  adult_raw <- bind_rows(
    read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
             col_names = col_names, trim_ws = TRUE,
             na = c("", "NA", "?"), show_col_types = FALSE) %>%
      mutate(split = "train"),
    read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
             col_names = col_names, skip = 1, trim_ws = TRUE,
             na = c("", "NA", "?"), show_col_types = FALSE) %>%
      mutate(split = "test")
  )

  adult_raw %>%
    mutate(income = str_remove(income, "\\.") %>% str_trim()) %>%
    clean_names() %>%
    mutate(
      across(c(workclass, education, marital_status, occupation,
               relationship, race, sex, native_country, income, split),
             as.factor),
      income = fct_relevel(income, ">50K"),
      age_group = case_when(
        age < 25 ~ "Jeune (<25)",
        age < 40 ~ "Adulte (25-39)",
        age < 55 ~ "Expérimenté (40-54)",
        TRUE     ~ "Senior (55+)"
      ) %>% factor(levels = c("Jeune (<25)", "Adulte (25-39)",
                               "Expérimenté (40-54)", "Senior (55+)")),
      capital_net   = capital_gain - capital_loss,
      high_hours    = factor(if_else(hours_per_week > 50, "Oui", "Non")),
      education_score = education_num,
      has_capital   = factor(if_else(capital_gain > 0 | capital_loss > 0, "Oui", "Non"))
    ) %>%
    drop_na()
}

# --- Chargement ou re-entraînement du modèle ---
load_model <- function(adult_fe) {
  if (file.exists("modele_fraude_final.rds")) {
    return(readRDS("modele_fraude_final.rds"))
  }

  # Re-entraînement rapide si le RDS est absent
  set.seed(2026)
  data_model <- adult_fe %>% filter(split == "train") %>% select(-split)
  splits_mod <- initial_split(data_model, prop = 0.8, strata = income)
  train_d    <- training(splits_mod)
  test_d     <- testing(splits_mod)
  folds_cv   <- vfold_cv(train_d, v = 5, strata = income)

  rec <- recipe(income ~ ., data = train_d) %>%
    step_rm(fnlwgt, education) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_zv(all_predictors()) %>%
    step_smote(income, over_ratio = 0.8)

  log_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
    set_engine("glmnet") %>% set_mode("classification")

  log_wf   <- workflow() %>% add_recipe(rec) %>% add_model(log_spec)
  log_grid <- grid_regular(penalty(range = c(-4, 0)), mixture(), levels = 5)

  log_res <- tune_grid(log_wf, resamples = folds_cv, grid = log_grid,
                       metrics = metric_set(roc_auc),
                       control = control_grid(verbose = FALSE))

  best_params <- select_best(log_res, metric = "roc_auc")
  final_wf    <- finalize_workflow(log_wf, best_params)
  final_fit   <- last_fit(final_wf, splits_mod)

  modele <- extract_workflow(final_fit)
  saveRDS(modele, "modele_fraude_final.rds")
  modele
}

# Chargement au démarrage
adult_fe <- load_data()
modele   <- load_model(adult_fe)

# KPIs globaux
n_total  <- nrow(adult_fe)
n_50k    <- sum(adult_fe$income == ">50K")
taux_50k <- n_50k / n_total

# AUC sur un échantillon test
set.seed(2026)
test_samp <- adult_fe %>% filter(split == "test") %>%
  sample_n(min(5000, sum(adult_fe$split == "test"))) %>%
  select(-split)
preds_glob <- predict(modele, test_samp, type = "prob") %>%
  bind_cols(test_samp %>% select(income))
auc_global <- roc_auc(preds_glob, truth = income, `.pred_>50K`)$.estimate

# ============================================================
# UI
# ============================================================

ui <- dashboardPage(
  skin = "blue",

  dashboardHeader(title = "Fraude Fiscale — Dashboard"),

  dashboardSidebar(
    sidebarMenu(
      id = "sidebar",
      menuItem("Tableau de bord",    tabName = "dashboard",   icon = icon("chart-bar")),
      menuItem("Exploration",        tabName = "exploration",  icon = icon("search")),
      menuItem("Scoring individuel", tabName = "scoring",      icon = icon("user-check"))
    ),

    # Filtres globaux (sidebar)
    hr(),
    h5("Filtres globaux", style = "padding-left:15px; color:#ccc;"),
    selectInput("filtre_workclass", "Classe de travail",
                choices = c("Toutes", levels(adult_fe$workclass)), selected = "Toutes"),
    selectInput("filtre_sex", "Sexe",
                choices = c("Tous", levels(adult_fe$sex)), selected = "Tous"),
    selectInput("filtre_age_group", "Tranche d'âge",
                choices = c("Toutes", levels(adult_fe$age_group)), selected = "Toutes"),
    selectInput("filtre_occupation", "Profession",
                choices = c("Toutes", levels(adult_fe$occupation)), selected = "Toutes")
  ),

  dashboardBody(
    tags$head(tags$style(HTML("
      .kpi-box { border-radius:8px; padding:15px; text-align:center; margin-bottom:15px; }
      .gauge-container { text-align:center; margin-top:20px; }
      .gauge-label { font-size:24px; font-weight:bold; padding:15px; border-radius:8px; }
      .faible   { background:#27AE60; color:white; }
      .moyen    { background:#F39C12; color:white; }
      .eleve    { background:#E74C3C; color:white; }
    "))),

    tabItems(

      # ---- ONGLET 1 : Tableau de bord ----
      tabItem(
        tabName = "dashboard",
        h2("Tableau de bord de surveillance"),

        # KPIs
        fluidRow(
          valueBoxOutput("kpi_total",   width = 3),
          valueBoxOutput("kpi_50k",     width = 3),
          valueBoxOutput("kpi_taux",    width = 3),
          valueBoxOutput("kpi_auc",     width = 3)
        ),

        fluidRow(
          box(
            title = "Taux de risque par profession", width = 6,
            status = "primary", solidHeader = TRUE,
            plotlyOutput("plot_occupation", height = 350)
          ),
          box(
            title = "Distribution par tranche d'âge", width = 6,
            status = "primary", solidHeader = TRUE,
            plotlyOutput("plot_age_group", height = 350)
          )
        ),

        fluidRow(
          box(
            title = "Heures travaillées vs revenu", width = 6,
            status = "info", solidHeader = TRUE,
            plotlyOutput("plot_heures", height = 300)
          ),
          box(
            title = "Capital net par classe", width = 6,
            status = "info", solidHeader = TRUE,
            plotlyOutput("plot_capital", height = 300)
          )
        )
      ),

      # ---- ONGLET 2 : Exploration ----
      tabItem(
        tabName = "exploration",
        h2("Exploration dynamique des données"),

        fluidRow(
          box(
            title = "Paramètres du graphique", width = 3, status = "warning",
            solidHeader = TRUE,
            selectInput("var_x", "Variable X",
                        choices = c("age", "education_num", "hours_per_week",
                                    "capital_gain", "capital_loss", "capital_net"),
                        selected = "age"),
            selectInput("var_color", "Couleur par",
                        choices = c("income", "sex", "age_group",
                                    "high_hours", "has_capital"),
                        selected = "income"),
            selectInput("type_graph", "Type de graphique",
                        choices = c("Histogramme", "Densité", "Boxplot"),
                        selected = "Histogramme"),
            sliderInput("n_sample", "Échantillon (n)",
                        min = 1000, max = 20000, value = 5000, step = 500)
          ),
          box(
            title = "Graphique dynamique", width = 9, status = "primary",
            solidHeader = TRUE,
            plotlyOutput("plot_dynamique", height = 400)
          )
        ),

        fluidRow(
          box(
            title = "Table des données — filtrable & exportable",
            width = 12, status = "primary", solidHeader = TRUE,
            DTOutput("table_donnees")
          )
        )
      ),

      # ---- ONGLET 3 : Scoring individuel ----
      tabItem(
        tabName = "scoring",
        h2("Scoring individuel en temps réel"),

        fluidRow(
          box(
            title = "Saisie du profil contribuable", width = 5,
            status = "warning", solidHeader = TRUE,

            sliderInput("s_age", "Âge", 18, 90, 40),
            selectInput("s_workclass", "Classe de travail",
                        choices = levels(adult_fe$workclass),
                        selected = "Private"),
            selectInput("s_education", "Niveau d'éducation",
                        choices = levels(adult_fe$education),
                        selected = "Bachelors"),
            selectInput("s_marital", "Statut marital",
                        choices = levels(adult_fe$marital_status),
                        selected = "Married-civ-spouse"),
            selectInput("s_occupation", "Profession",
                        choices = levels(adult_fe$occupation),
                        selected = "Exec-managerial"),
            selectInput("s_relationship", "Relation",
                        choices = levels(adult_fe$relationship),
                        selected = "Husband"),
            selectInput("s_race", "Origine ethnique",
                        choices = levels(adult_fe$race),
                        selected = "White"),
            selectInput("s_sex", "Sexe",
                        choices = levels(adult_fe$sex),
                        selected = "Male"),
            numericInput("s_capital_gain",  "Capital gain ($)",  0, min = 0),
            numericInput("s_capital_loss",  "Capital loss ($)",  0, min = 0),
            sliderInput("s_hours", "Heures/semaine", 1, 99, 40),
            selectInput("s_country", "Pays d'origine",
                        choices = levels(adult_fe$native_country),
                        selected = "United-States"),
            actionButton("btn_score", "Calculer le score",
                         class = "btn-primary btn-lg btn-block",
                         icon = icon("calculator"))
          ),

          box(
            title = "Résultat du scoring", width = 7,
            status = "danger", solidHeader = TRUE,

            div(class = "gauge-container",
                h4("Score de risque de sous-déclaration"),
                br(),
                uiOutput("score_gauge"),
                br(),
                uiOutput("score_interpretation"),
                br(),
                plotlyOutput("score_gauge_plot", height = 250)
            )
          )
        )
      )
    )
  )
)

# ============================================================
# SERVER
# ============================================================

server <- function(input, output, session) {

  # ---- Données filtrées (réactive) ----
  data_filtered <- reactive({
    df <- adult_fe

    if (input$filtre_workclass != "Toutes")
      df <- df %>% filter(workclass == input$filtre_workclass)
    if (input$filtre_sex != "Tous")
      df <- df %>% filter(sex == input$filtre_sex)
    if (input$filtre_age_group != "Toutes")
      df <- df %>% filter(age_group == input$filtre_age_group)
    if (input$filtre_occupation != "Toutes")
      df <- df %>% filter(occupation == input$filtre_occupation)

    df
  })

  # ---- KPIs ----
  output$kpi_total <- renderValueBox({
    valueBox(
      format(nrow(data_filtered()), big.mark = " "),
      "Total individus",
      icon = icon("users"), color = "blue"
    )
  })

  output$kpi_50k <- renderValueBox({
    n <- sum(data_filtered()$income == ">50K")
    valueBox(
      format(n, big.mark = " "),
      "Individus >50K$",
      icon = icon("exclamation-triangle"), color = "red"
    )
  })

  output$kpi_taux <- renderValueBox({
    taux <- mean(data_filtered()$income == ">50K")
    valueBox(
      percent(taux, accuracy = 0.1),
      "Taux de risque",
      icon = icon("percent"), color = "orange"
    )
  })

  output$kpi_auc <- renderValueBox({
    valueBox(
      round(auc_global, 3),
      "AUC-ROC modèle",
      icon = icon("chart-line"), color = "green"
    )
  })

  # ---- Graphiques Dashboard ----
  output$plot_occupation <- renderPlotly({
    df <- data_filtered() %>%
      group_by(occupation) %>%
      summarise(taux = mean(income == ">50K"), n = n()) %>%
      filter(n > 30) %>%
      arrange(desc(taux))

    plot_ly(df, x = ~taux, y = ~reorder(occupation, taux),
            type = "bar", orientation = "h",
            marker = list(color = ~taux, colorscale = "RdYlGn",
                          reversescale = TRUE)) %>%
      layout(xaxis = list(title = "Taux de risque", tickformat = ".0%"),
             yaxis = list(title = ""),
             margin = list(l = 120))
  })

  output$plot_age_group <- renderPlotly({
    df <- data_filtered() %>%
      count(age_group, income) %>%
      group_by(age_group) %>%
      mutate(pct = n / sum(n))

    plot_ly(df %>% filter(income == ">50K"),
            x = ~age_group, y = ~pct, type = "bar",
            marker = list(color = "#E74C3C")) %>%
      layout(xaxis = list(title = "Tranche d'âge"),
             yaxis = list(title = "Taux >50K$", tickformat = ".0%"))
  })

  output$plot_heures <- renderPlotly({
    df <- data_filtered()
    plot_ly(df, x = ~hours_per_week, color = ~income,
            type = "histogram", alpha = 0.6, nbinsx = 30,
            colors = c("#E74C3C", "#3498DB")) %>%
      layout(xaxis = list(title = "Heures/semaine"),
             yaxis = list(title = "Effectif"),
             barmode = "overlay")
  })

  output$plot_capital <- renderPlotly({
    df <- data_filtered() %>% filter(abs(capital_net) > 0)
    plot_ly(df, y = ~capital_net, color = ~income, type = "box",
            colors = c("#E74C3C", "#3498DB")) %>%
      layout(yaxis = list(title = "Capital net ($)", type = "log"),
             showlegend = TRUE)
  })

  # ---- Exploration dynamique ----
  output$plot_dynamique <- renderPlotly({
    df <- data_filtered() %>%
      sample_n(min(input$n_sample, nrow(data_filtered())))

    var_x   <- input$var_x
    var_col <- input$var_color

    p <- switch(input$type_graph,
      "Histogramme" = ggplot(df, aes_string(x = var_x, fill = var_col)) +
        geom_histogram(bins = 35, position = "dodge", alpha = 0.8) +
        scale_fill_brewer(palette = "Set1"),
      "Densité" = ggplot(df, aes_string(x = var_x, fill = var_col)) +
        geom_density(alpha = 0.5) +
        scale_fill_brewer(palette = "Set1"),
      "Boxplot" = ggplot(df, aes_string(x = var_col, y = var_x, fill = var_col)) +
        geom_boxplot(alpha = 0.7, show.legend = FALSE) +
        scale_fill_brewer(palette = "Set1")
    )

    p <- p + theme_minimal(base_size = 12) +
      labs(x = var_x, fill = var_col)

    ggplotly(p)
  })

  output$table_donnees <- renderDT({
    data_filtered() %>%
      select(age, workclass, education, occupation, sex,
             hours_per_week, capital_gain, capital_loss,
             capital_net, income, age_group, high_hours) %>%
      datatable(
        filter   = "top",
        rownames = FALSE,
        extensions = "Buttons",
        options  = list(
          pageLength = 15,
          scrollX    = TRUE,
          dom        = "Bfrtip",
          buttons    = c("copy", "csv", "excel", "pdf", "print")
        )
      )
  })

  # ---- Scoring individuel ----
  score_result <- eventReactive(input$btn_score, {
    # Construire le profil
    nouveau <- tibble(
      age             = as.integer(input$s_age),
      workclass       = factor(input$s_workclass, levels = levels(adult_fe$workclass)),
      fnlwgt          = 200000L,
      education       = factor(input$s_education, levels = levels(adult_fe$education)),
      education_num   = as.integer(which(levels(adult_fe$education) == input$s_education)),
      marital_status  = factor(input$s_marital,   levels = levels(adult_fe$marital_status)),
      occupation      = factor(input$s_occupation, levels = levels(adult_fe$occupation)),
      relationship    = factor(input$s_relationship, levels = levels(adult_fe$relationship)),
      race            = factor(input$s_race,       levels = levels(adult_fe$race)),
      sex             = factor(input$s_sex,        levels = levels(adult_fe$sex)),
      capital_gain    = as.integer(input$s_capital_gain),
      capital_loss    = as.integer(input$s_capital_loss),
      hours_per_week  = as.integer(input$s_hours),
      native_country  = factor(input$s_country,   levels = levels(adult_fe$native_country)),
      income          = factor(">50K",             levels = levels(adult_fe$income)),
      age_group       = factor(
        case_when(
          input$s_age < 25 ~ "Jeune (<25)",
          input$s_age < 40 ~ "Adulte (25-39)",
          input$s_age < 55 ~ "Expérimenté (40-54)",
          TRUE             ~ "Senior (55+)"
        ),
        levels = c("Jeune (<25)", "Adulte (25-39)", "Expérimenté (40-54)", "Senior (55+)")
      ),
      capital_net     = as.integer(input$s_capital_gain - input$s_capital_loss),
      high_hours      = factor(if_else(input$s_hours > 50, "Oui", "Non")),
      education_score = as.integer(which(levels(adult_fe$education) == input$s_education)),
      has_capital     = factor(
        if_else(input$s_capital_gain > 0 | input$s_capital_loss > 0, "Oui", "Non")
      ),
      split           = factor("test")
    )

    pred <- predict(modele, nouveau %>% select(-split), type = "prob")
    list(
      score_risque = pred[[".pred_>50K"]],
      score_safe   = pred[[".pred_<=50K"]]
    )
  })

  output$score_gauge <- renderUI({
    req(score_result())
    s <- score_result()$score_risque
    niveau <- if (s < 0.3) "FAIBLE" else if (s < 0.6) "MOYEN" else "ÉLEVÉ"
    classe <- tolower(niveau) %>% str_remove_all("É|Ê")
    classe <- if (s < 0.3) "faible" else if (s < 0.6) "moyen" else "eleve"

    div(
      class = paste("gauge-label", classe),
      sprintf("Score : %.1f%% — Risque %s", s * 100, niveau)
    )
  })

  output$score_interpretation <- renderUI({
    req(score_result())
    s <- score_result()$score_risque
    msg <- if (s < 0.3) {
      "✅ Ce profil présente un faible risque de sous-déclaration fiscale."
    } else if (s < 0.6) {
      "⚠️ Ce profil mérite une vérification approfondie."
    } else {
      "🔴 Risque élevé — audit fiscal recommandé."
    }
    p(msg, style = "font-size:16px; margin-top:10px;")
  })

  output$score_gauge_plot <- renderPlotly({
    req(score_result())
    s <- round(score_result()$score_risque * 100, 1)
    couleur <- if (s < 30) "#27AE60" else if (s < 60) "#F39C12" else "#E74C3C"

    plot_ly(
      type  = "indicator",
      mode  = "gauge+number",
      value = s,
      number = list(suffix = "%"),
      gauge = list(
        axis  = list(range = list(0, 100)),
        bar   = list(color = couleur),
        steps = list(
          list(range = c(0, 30),  color = "#D5F5E3"),
          list(range = c(30, 60), color = "#FDEBD0"),
          list(range = c(60, 100), color = "#FADBD8")
        ),
        threshold = list(
          line  = list(color = "black", width = 3),
          thickness = 0.75,
          value = s
        )
      )
    ) %>%
      layout(margin = list(t = 30, b = 10))
  })
}

# ============================================================
# LANCEMENT
# ============================================================
shinyApp(ui = ui, server = server)
