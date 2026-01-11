# Projet-Prédiction-matchs-saison-24-25-ligue-1-M1
### Packages utilisés
```{r}
library(tidyr)
library(tidyverse)
library(lubridate)
library(dplyr)
library(readr)
library(stringr)
library(slider)
library(nnet)
library(shiny)
library(shinythemes)
library(data.table)
```
# Introduction

## 1) Nettoyage des données
### (i) Importation des jeux de données de toutes les saisons.

```{r}
df_2021_2022 <- read_csv("C:/Users/Daniel/OneDrive/MPE/Analyse de donnée/Saison 2021-2022.csv")
df_2022_2023 <- read_csv("C:/Users/Daniel/OneDrive/MPE/Analyse de donnée/Saison 2022-2023.csv")
df_2023_2024 <- read_csv("C:/Users/Daniel/OneDrive/MPE/Analyse de donnée/Saison 2023-2024.csv")
df_2024_2025 <- read_csv("C:/Users/Daniel/OneDrive/MPE/Analyse de donnée/Saison 2024-2025.csv")
df_2025_2026 <- read_csv("C:/Users/Daniel/OneDrive/MPE/Analyse de donnée/Saison 2025-2026.csv")

```

### (ii) Création de de la base de données finale

```{r}
matches <- bind_rows(
  df_2021_2022,
  df_2022_2023,
  df_2023_2024,
  df_2024_2025,
  df_2025_2026) %>%
  arrange(Date)
matches <- matches %>%
  select(
    Date,
    HomeTeam, AwayTeam,
    FTR, FTHG, FTAG,
    HS,AS,HST,AST,HC,
    AC,HY,AY,HR,AR)
```

### (iii) Restructuration des dates
On s'assure que la date est bien au format adéquat mais cela n'est pas le cas donc on est obligé de transformer les dates sous forme --/--/---- en ..-..-.... .

```{r}
library(lubridate)                                                              #lubridate permet l'utilisation d'outils pour nettoyer les dates (dmy, ymd, year, month...).

matches <- matches %>%                                                          #On convertit la colonne Date en vrai type Date (R) à partir d'un texte au format "jour-mois-année".
  mutate(Date = dmy(Date))                                                      #dmy() signifie Day-Month-Year ex: 15/09/2023 devient 2023-09-15.
```

### (iv) Saisonnalité
On crée la variable "season" et on trie de manière croissante afin de respecter un ordre chronologique cohérent pour le modèle, ce qui est fondamental en machine learning temporel.

```{r}
matches <- matches %>%                                                          #On ajoute une nouvelle variable appelée "season" au jeu de données matches.
  mutate(                                         
    season = if_else(                                                           #La saison de football commence en juillet (mois 7).
      month(Date) >= 7,                                                         #Si le mois du match est >= juillet,
      paste0(year(Date), "-", year(Date) + 1),                                  #la saison est "année-année+1" (ex : 2023-2024).
      paste0(year(Date) - 1, "-", year(Date))                                   #Match joué entre janvier et juin.
    )
  )

matches <- matches %>%
  arrange(Date)                                   #
```

## 2)  Introduction des variables, de l'objectif et des moyens à mettre en oeuvre pour l'atteindre.
Afin d'entraîner notre modèle statistique d'apprentissage, il est nécessaire de s'accomoder la base de données et ses variables natives. Effectivement, notre objectif étant la prédiction de probabilités pour chaque issue d'un match de football en ligue 1 sur la saison 2025-2026, la précision maximale pour chaque issue est la plus souhaitable. Afin de converger vers cela, nous devons dans un premier temps créer des "features" c'est-à-dire un nombre nécessaire et suffisant de variables qui va nous permettre de créer nos "features" principales, à savoir celles qui captent une réelle dynamique de domination de telle ou telle équipe sur son adversaire lors d'un match. On peut mentionner qu'avant même d'importer la base de données, nous avons nettoyés et sélectionnés les variables les plus pertinantes quant aux objectifs de features fixés (NB : On a sélectionnés plus de variables qu'il n'en fallait par simple sécurité) . Nous allons utilisés dans un premier temps les variables indépendantes natives : *FTR*, *HomeTeam*, *AwayTeam* et *Date* qui représentent respectivement le résultat d'un match (victoire, défaite ou égalité), le nom de l'équipe à domicile, le nom de l'équipe extérieur et la date du match. Grâce à *FTR* nous allons pouvoir créer nos deux premières features qui vont nous permettre de créer nos features principaux, respectivement les points gagnés par l'équipe à domicile et par l'équipe extérieur lors d'un match (*home_points* et *away_points*). De plus, nous allons également définir une variable *match_id* par souci de numérisation claire du nombre de match dans la base de données. Cela va nous permettre d'avoir un historique limpide pour chaque match et chaque équipe. Après avoir réalisé tout cela, nous pouvons créer dans un premier temps par cheminement logique les différentiels suivants :
- Celui de la forme récente entre les deux équipes qui s'affrontent lors d'un match
- Celui des points cumulés entre les deux équipes qui s'affrontent lors d'un match
- Celui des buts cumulés (différence entre les buts marqués et encaissés) de l'équipe domicile et extérieur avant le match.

Après cela, nous nous attarderons sur la performance d'une modélisation par apprentissage statistique, grâce à l'entraînement de nos features sur les trois saisons précédentes, en l'évaluant sur la saison 2024-2025. A l'aide d'un modèle multinomial on va déterminer la précision de ce dernier à prédire les probabilités d'issues d'un match de football. 
Enfin, on va essayé de prédire les résultats de match, du moins leurs probabilités, pour les matchs à venir sur la saison actuelle 2025-2026 (matchs d'après trêve).
On concluera ensuite quant aux résultats de notre modèle tout en énoncant ses différentes limites.

# I) Phase 1 : Entraînement (apprentissage) sur les données des saisons 2021-2022, 2022-2023 et 2023-2024 réalisé notamment grâce à la création de features 

## 1) Création des nouvelles variables 
On code les features principales :
-   Forme récente (mesurer la dynamique sur les 5 derniers matchs)
-   Points cumulés (performance globale sur l'ensemble de la saison)
-   Différentiels de buts (représente la qualité globale si l'équipe
    marque beaucoup et/ou encaisse peu ou pas).

Afin de procéder à cela, on prépare les variables cibles et explicatives nécessaires à l'entraînement du modèle : points gagnés par l'équipe à domicile (home_points), points gagnés par l'équipe à l'extérieur (away_points) et un identifiant unique par match (match_id).

H signifie victoire pour l'équipe à domicile, A pour l'équipe à l'extérieur et D égalité pour les deux équipes.

```{r}
matches <- matches %>%
  arrange(Date) %>%
  mutate(
    home_points = case_when(                                                    #Attribution des points pour l'équipe à domicile.
      FTR == "H" ~ 3,                                                           #Victoire pour l'équipe à domicile et donc rajout de 3 points à cette équipe dans le classement.
      FTR == "D" ~ 1,                                                           #Match nul donc gain d'un point.
      TRUE ~ 0                                                                  #Dans le dernier cas (défaite), aucun gain.
    ),
    away_points = case_when(                                                    #Attribution des points pour l'équipe à l'extérieur.
      FTR == "A" ~ 3,                                                           #Victoire pour l'équipe à l'extérieur et donc rajout de 3 points à cette équipe dans le classement.
      FTR == "D" ~ 1,                                                           #Match nul donc gain d'un point.
      TRUE ~ 0                                                                  #Dans le dernier cas (défaite), aucun gain.
    ),
    match_id = row_number()                                                     #Création d'un identifiant unique pour chaque match avec row_number() qui suit l'ordre chronologique défini par arrange(Date)                                                                             
  )
```

Ce code, jusqu'à-là, nous permet de garantir que le modèle n'utilise jamais des informations du futur. De plus, nous avons les points qui correspondent exactement aux règles officielles (3-1-0). Ces variables serviront ensuite à calculer la forme récente, les points cumulés et les différentiels de buts. La variable match_id permet de relier facilement un match à ses features et prédictions.

## 2) Forme récente
- Pour chaque match, on calcule la performance d'une équipe sur ses N derniers matchs avant ce match (ex: N = 5) 
- On décale (lag) d'un match, ainsi le match du jour ne doit jamais rentrer dans sa propre forme. 

Pour chaque équipe avant le match on veut établir la moyenne des points sur les 5 derniers matchs et déterminer quelle équipe est favorite (différence de forme). L'objectif du code ci-dessous est de créer trois variables qui vont matérialiser la forme récente. 

Le code se décompose en trois étapes pour le premier feature : 
(i) La première étape est d'avoir une ligne par équipe et par match au lieu d'une ligne de match avec l'équipe domicile et extérieur. Cela facilite les calculs d'historique par équipe. 
(ii) La deuxième étape est, pour chaque équipe, calculer la moyenne de points sur les 5 matchs précédents (forme récente). 
(iii) La troisième étape est la création de trois colonnes dans matches, à savoir la forme récente de l'équipe domicile, de l'équipe extérieur et la différence de forme entre les deux équipes du match.

```{r}
# Première Étape : 

N <- 5                                                                          #Nombre de matchs précédents utilisés pour mesurer la "forme récente".

team_matches <- matches %>%
  arrange(Date) %>% 
  select(match_id, Date, HomeTeam, AwayTeam, home_points, away_points) %>%
  pivot_longer(                                                                 #pivot_longer() : transforme des colonnes en lignes (passage du format large au long).
    cols = c(HomeTeam, AwayTeam),                                               #On va créer 2 lignes par match, une ligne pour HomeTeam et une autre pour AwayTeam.
    names_to  = "side",                                                         #names_to est le nom de la nouvelle colonne qui indiquera l'origine
                                                                                #side prendra comme valeur "HomeTeam" ou "AwayTeam"
    values_to = "team"                                                          #values_to est le nom de la nouvelle colonne qui contiendra le nom de l'équipe
  ) %>%
  mutate(
    points = if_else(side == "HomeTeam", home_points, away_points)              #Si l'équipe est HomeTeam alors on prend home_points sinon on prend away_points,
  ) %>%                                                                         #On a une seule colonne "points" utilisable pour toutes les équipes.
  select(match_id, Date, team, points)

#Deuxième Étape :

team_form <- team_matches %>%
  group_by(team) %>%
  mutate(
    form_pts_lastN = slide_dbl(                                                 #Calcule une statistique sur une fenêtre glissante (rolling window) et renvoie un vecteur numérique.(dbl).
      
      .x = lag(points),                                                         #.x est la série sur laquelle on calcule le rolling.
                                                                                #lag(points) veut dire qu'on décale d'une ligne vers le bas (on exclut le match actuel).
                                                                                #Le match du jour ne rentre pas dans sa propre forme et permet l'anti-fuite.
      
      .f = ~ mean(.x, na.rm = TRUE),                                            #.f est une fonction appliquée sur la fenêtre.
                                                                                #Ici on a la moyenne des points sur la fenêtre.
                                                                                #na.rm = TRUE va ignorer les NA en début d'historique.
                                                                                
      
      .before = N - 1,                                                          #.before est la taille de la fenêtre vers le passé
                                                                                #N - 1 + l'observation courante (après lag) = N matchs au total
      
      .complete = FALSE                                                         #.complete = FALSE veut dire qu'on autorise des fenêtres incomplètes (moins de N matchs),
                                                                                #Mais si l'équipe n'a aucun match précédent, la forme reste NA/NaN (pas d'historique)
                                                                                
    )
  ) %>%
  
  ungroup() %>%                                                                 #ungroup() permet de retirer le group_by (bonne pratique pour éviter des effets de bord) 
                                                                                #Un effet de bord c'est lorsqu'on a fait un regroupement sur une ou des colonne(s) et qu'après                                                                                        #avoir fait ce qu'on avait à faire pour cette colonne, les fonctions d'après sont toujours appliquées                                                                                 #à cette colonne alors que ce n'est pas le but.
  select(match_id, team, form_pts_lastN)                                        #On garde seulement les colonnes nécessaires au join final
                                                                                
#Troisième Étape : 

matches <- matches %>%                                                          #Join pour l'équipe à domicile.
  left_join(
    team_form %>% rename(home_form_pts_lastN = form_pts_lastN),
    by = c("match_id", "HomeTeam" = "team")
  )

matches <- matches %>%                                                          #Join pour l'équipe à l'extérieur.
  left_join(
    team_form %>% rename(away_form_pts_lastN = form_pts_lastN),
    by = c("match_id", "AwayTeam" = "team")
  )

matches <- matches %>%                                                          #Différence de forme (domicile - extérieur).
  mutate(form_diff_lastN = home_form_pts_lastN - away_form_pts_lastN)
```

3 nouvelles variables créées qui représentent respectivement : 
- La moyenne des points pour l'équipe à domicile sur ses 5 matchs précédents avant le match actuel (home_form_pts_lastN).
- La moyenne des points pour l'équipe extérieur sur ses 5 matchs précédents avant le match actuel (away_form_pts_lastN).
- Différence de forme (form_diff_lastN).

On va uniquement garder *form_diff_lastN*. En effet, cela ne sert à rien d'intégrer les deux autres variables singulièrement car la différence de forme est endogène aux moyennes de points pour les deux équipes sur les 5 matchs précédents avant le match actuel.

## 3) Points cumulés

On va donc procéder au calcul des points cumulés par équipe et par saison. L'objectif est de calculer pour chaque équipe le total de points cumulés avant chaque match et la différence de points cumulés puis faire une jointure dans la base de données finale matches.

Comme pour la forme récente on va procéder en trois étapes :
(i) Mise au format long c'est-à-dire une ligne par équipe et par match.
(ii) Calcul des points cumulés avant le match.
(iii) Jointure dans la base de données finale.

```{r}
#Première Étape : 

team_points <- matches %>%
  select(match_id, Date, season, HomeTeam, AwayTeam, home_points, away_points) %>%
  pivot_longer(                                                                 #Transformation du format "large" en format "long".
    cols = c(HomeTeam, AwayTeam),                                               #Chaque match génère deux lignes : une pour HomeTeam et une pour AwayTeam.
    names_to = "side",                                                          #Indique si l'équipe joue à domicile ou à l'extérieur.
    values_to = "team"                                                          #Nom de l'équipe.
  ) %>%
  mutate(
    points = if_else(side == "HomeTeam", home_points, away_points)              #Création d'une variable unique "points", elle prend la valeur home_points ou away_points selon le                                                                                    #rôle de l'équipe.
  ) %>%
  select(match_id, Date, season, team, points)                                  #On conserve uniquement les colonnes utiles pour le calcul des cumuls.

#Deuxième Étape : 

team_cum_pts <- team_points %>%
  group_by(team, season) %>%                                                    #Regroupement par équipe et par saison, le cumul repart donc à zéro au début de chaque saison.
  mutate(
    cum_pts_before = lag(cumsum(points), default = 0)                           #cumsum(points) calcule la somme cumulée incluant le match courant
                                                                                #lag(...) décale cette somme d'un match, on obtient ainsi le nombre de points cumulés avant le match                                                                                  #courant.
   ) %>%
  ungroup() %>%
  select(match_id, team, season, cum_pts_before)

#Troisième Étape : 

matches <- matches %>%                                                          #Raisonemment identique dans le code que la forme récente (dans la structure).
  left_join(
    team_cum_pts %>% rename(home_cum_pts = cum_pts_before),
    by = c("match_id", "season", "HomeTeam" = "team")
  )

matches <- matches %>%
  left_join(
    team_cum_pts %>% rename(away_cum_pts = cum_pts_before),
    by = c("match_id", "season", "AwayTeam" = "team")
  )

matches <- matches %>%
  mutate(
    cum_pts_diff = home_cum_pts - away_cum_pts
  )
```

On a donc créée trois nouvelles variables :

- Les points cumulés de l'équipe à domicile avant le match (home_cum_points).
- Les points cumulés de l'équipe à l'extérieur avant le match (away_cum_pts).
- L'avantage en points cumulés (cum_pts_diff), qui est la différence entre les points cumulés de l'équipe à domicile (home_cum_pts) et extérieur (away_cum_pts) avant le match.

On va seulement garder *cum_pts_diff* pour l'évaluation du modèle entraîné.

## 4) Différentiels de buts

L'objectif de cette sous-partie est de calculer pour chaque équipe et chaque saison le cumul de buts marqués/encaissés avant chaque match mais également de rattacher ces cumuls au match et créer des variables de comparaison entre les deux équipes qui s'affrontent pour un match donné.

On a 4 étapes à suivre :
(i) Mise au format long et création de variables
(ii) Cumuls avant le match
(iii) Jointure dans la base de données principale
(iv) Créer les différentiels et jointure dans matches

```{r}

#Première Étape : 

team_goals <- matches %>%
  select(match_id, Date, season, HomeTeam, AwayTeam, FTHG, FTAG) %>%
  
  pivot_longer(
    cols = c(HomeTeam, AwayTeam),
    names_to  = "side",   #
    values_to = "team"    
  ) %>%
  
  mutate(
    goals_for = if_else(side == "HomeTeam", FTHG, FTAG),                        #Buts marqués (goals_for)
    goals_against = if_else(side == "HomeTeam", FTAG, FTHG),                    #Buts encaissés (goals_against), les deux dépendent du fait d'être à domicile ou à l'extérieur.
    across(c(goals_for, goals_against), ~ replace_na(.x, 0))                    #On remplace les deux seuls NA dans team_goals car sinon les NA se propagent ensuite dans tous les                                                                                    #cumuls de matches.
    ) %>%                                                                       
  
  select(match_id, Date, season, team, goals_for, goals_against)

#Deuxième Étape :

team_cum_goals <- team_goals %>%
  group_by(team, season) %>%                                                    #Reset au début de chaque saison
  mutate(
    cum_gf_including = cumsum(goals_for),                                       #Addition des buts marqués match après match en incluant le match courant. On ne peut pas utiliser                                                                                    #tel quel.
    cum_ga_including = cumsum(goals_against),                                   #Addition des buts encaissés match après match en incluant le match courant. On ne peut pas utiliser                                                                                  #tel quel.
    cum_gf_before = lag(cum_gf_including, default = 0),                         #On décale d'un match pour obtenir le cumul avant le match courant
    cum_ga_before = lag(cum_ga_including, default = 0),                         #default = 0 : pour le premier match de la saison on considère que l'équipe a 0 but cumulé avant de                                                                                   #jouer.
    cum_gd_before = cum_gf_before - cum_ga_before                               #Différence de buts cumulée avant le match = buts marqués - buts encaissés
  ) %>%                                                                         #Différence positive veut dire que l'équipe est perfomante
                                                                                #Différence négative veut dire que l'équipe est en difficulté
  ungroup() %>%
  select(match_id, team, season, cum_gf_before, cum_ga_before, cum_gd_before)   
                                                                                
#Troisième Étape : 

matches <- matches %>%
  left_join(
    team_cum_goals %>%
      rename(
        home_cum_gf = cum_gf_before,
        home_cum_ga = cum_ga_before,
        home_cum_gd = cum_gd_before
      ),
    by = c("match_id", "season", "HomeTeam" = "team")
  )

matches <- matches %>%
  left_join(
    team_cum_goals %>%
      rename(
        away_cum_gf = cum_gf_before,
        away_cum_ga = cum_ga_before,
        away_cum_gd = cum_gd_before
      ),
    by = c("match_id", "season", "AwayTeam" = "team")
  )

#Quatrième Étape :

matches <- matches %>%
  mutate(
    cum_gf_diff = home_cum_gf - away_cum_gf,                                    #Différence entre les buts marqués cumulés par l'équipe à domicile et l'équipe extérieur, cela mesure                                                                                 #la puissance offensive relative des deux équipes qui s'affrontent au match courant. 
                                                                                #Si cum_gf_diff > 0, l'équipe à domicile a marqué plus de buts que celle extérieur depuis le début de                                                                                 #la saison. 
                                                                                #Si cum_gf_diff < 0, l'équipe extérieur a marqué plus de buts que celle à domicile depuis le début de                                                                                 #la saison.
    
    cum_ga_diff = home_cum_ga - away_cum_ga,                                    #Différence entre les buts encaissés cumulés par l'équipe à docimile et l'équipe extérieure, moins on                                                                                 #encaisse mieux c'est donc le signe est contre-intuitif.                                                     
                                                                                #Si cum_ga_diff > 0, l'équipe domicile a un désavantage défensif. Elle a encaissé plus de but que                                                                                     #l'équipe extérieur.
                                                                                #Si cum_ga_diff < 0, l'équipe domicile a un avantage défensif. Elle a encaissé moins de but que                                                                                       #l'équipe extérieur.
    cum_gd_diff = home_cum_gd - away_cum_gd                                     #La différence entre les buts cumulés de l'équipe domicile et extérieur.
  )                                                                             #cum_gd_diff > 0, l'équipe domicile est globalement supérieur à l'équipe extérieur (avantage net).
                                                                                #cum_gd_diff < 0, l'équipe domicile est globalement inférieur à l'équilibre extérieur (avantage net).
```

On a donc créée trois nouvelles variables sur lesquelles on va se concentrer :
- cum_gf_diff qui mesure l'attaque.
- cum_ga_diff qui mesure la défense.
- cum_gd_diff qui mesure la performance globale des équipes.

Nos trois features principales qu'on va utiliser pour l'évaluation du modèle entraîné sur la saison 2024-2025 sont :

- *form_diff_lastN* qui représente la différence de forme récente entre les deux équipes avant qu'elles s'affrontent.
- *cum_pts_diff* qui représente la différence entre les points cumulés de l'équipe à domicile (home_cum_pts) et extérieur (away_cum_pts) avant le match.
- *cum_gd_diff* qui représente la différence entre les buts cumulés de l'équipe domicile et extérieur avant le match.

Il est important de préciser que *cum_pts_diff* et *cum_gd_diff* sont des features reinitiallisées à chaque nouvelle saison. En effet, il serait illogique de conserver les performances individuelles des équipes d'une saison à l'autre car cela pourrait grandement biaiser notre modèle avec une contamination temporelle structurelle qui peut potentiellement causer une propagation du passé vers le futur, biaiser les features et gonfler artificiellement les performances hors-échantillon. De plus, il serait impertinent de laisser dans notre modèle des variables qui sont des combinaisons linéaires parfaites de nos features principales c'est pour cela qu'on sélectionne les différences qui reflètent massivement la dynamique et les rapports de force entre les équipes au moment d'un match entre deux équipes de la saison considérée.

# II) Phase 2 : Évaluation du modèle entraîné sur la saison 2024–2025

## 1) Application du modèle entraîné sur la saison 2024-2025

Les données sont séparées en :
- un échantillon d’apprentissage (saisons 2021–2022 à 2023–2024),
- un échantillon de test (saison 2024–2025),

afin d’évaluer les performances du modèle dans un cadre temporel réaliste et sans fuite d’information.

### (i) Préparation de la variable dépendante

```{r}
# Transformation de FTR en factor
matches <- matches %>%
  mutate(
    FTR = factor(FTR, levels = c("H", "D", "A")))
```

### (ii) Construction des échantillons d'apprentissage et de test

Les premières journées de chaque saison ne disposent pas d’information sur la forme récente des équipes, alors ces observations sont exclues.

```{r}
# Échantillon d'apprentissage
train_data <- matches %>%
  filter(season %in% c("2021-2022", "2022-2023", "2023-2024")) %>%
  drop_na(form_diff_lastN)

# Échantillon de test (saison 2024-2025)
test_data <- matches %>%
  filter(season == "2024-2025") %>%
  drop_na(form_diff_lastN)

```

### (iii) Entrainement du modèle de classification multinomiale

Le modèle est entraîné uniquement sur les saisons passées afin d’éviter toute fuite d’information temporelle.

```{r}
# Modèle multinomiale avec les features principaux
model_multinom <- multinom(
  FTR ~ form_diff_lastN + cum_pts_diff + cum_gd_diff,
  data = train_data,
  trace = FALSE)
```

### (iv) Prédiction des probabilités (saison 2024/2025)

Le modèle fournit, pour chaque match, une probabilité associée à chacune des trois issues possibles.

```{r}
proba_test <- predict(
  model_multinom,
  newdata = test_data,
  type = "probs")

# Intégration des probabilités dans la base de test
test_data <- test_data %>%
  mutate(
    proba_H = proba_test[, "H"],
    proba_D = proba_test[, "D"],
    proba_A = proba_test[, "A"])
```

## 2) Évaluation des performances du modèle

### (i) Classe prédite

La classe prédite correspond à l'issue dont la probabilité estimée par le modèle est la plus élevé (probabilité maximale parmi H,D,A pour chaque match)

```{r}
test_data <- test_data %>%
  mutate(
    pred_class = colnames(proba_test)[apply(proba_test, 1, which.max)])
```

### (ii) Matrice de confusion et accuracy globale du modèle

```{r}
confusion_matrix <- table(
  Observed = test_data$FTR,
  Predicted = test_data$pred_class)
confusion_matrix
```
On peut constater que le modèle prédit relativement bien les victoires domiciles, mais rencontre davantage de difficultés pour les victoires extérieures. Les matches nuls sont jamais prédits, ce qui reflète la complexité et la forte incertitude associées à ce type d’issue.

```{r}
accuracy <- mean(test_data$FTR == test_data$pred_class)
accuracy
```
Le modèle atteint une accuracy d’environ 54 % sur la saison 2024–2025, ce qui est significativement supérieur à une prédiction aléatoire (33 %).
Ce résultat met en évidence la capacité du modèle à capter une partie des dynamiques de performance des équipes, tout en soulignant le caractère aléatoire du football.

# III) Phase 3 : Prédiction des probabilités sur la saison 2025–2026 

## 1) Génération des probabilités de victoire pour les matchs déjà joués de la saison 25/26
L’objectif de cette phase est d’utiliser le modèle multinomial entraîné sur les saisons 2021–2022 à 2023–2024 et évalué sur la saison 2024–2025, afin de prédire les probabilités de victoire à domicile (H), de match nul (D) et de victoire à l’extérieur (A) pour tous les matches de la saison 2025–2026. 

Il s’agit de probabilités ex ante, fondées uniquement sur les informations disponibles avant chaque match : Forme récente, points cumulés, différentiel de buts cumulés. 
Nous faisons bien attention qu'aucune information issue des résultats de la saison 2025–2026 n’est utilisée dans l’apprentissage du modèle, garantissant l’absence totale de fuite temporelle ou de bias. 

### (i) Construction de l’échantillon de prédiction (saison 2025–2026) 
Comme précédemment, les premiers matches de la saison ne disposent pas d’historique suffisant pour calculer la forme récente. Ces observations sont donc exclues.

```{r}
# Échantillon de prédiction : saison 2025-2026
features <- c(
  "form_diff_lastN",
  "cum_pts_diff",
  "cum_gd_diff")
pred_2526 <- matches %>%
  filter(season == "2025-2026") %>%
  drop_na(features) %>%
  select(match_id, Date, HomeTeam, AwayTeam,FTR, features)
```

### (ii) Prédiction des probabilités avec le modèle entraîné 
On applique le modèle multinomial déjà estimé à ces nouvelles observations.

```{r}
train_data_final <- matches %>%
  filter(season %in% c("2021-2022", "2022-2023", "2023-2024","2024-2025")) %>%
  drop_na(form_diff_lastN)

model_multinom <- multinom(
  FTR ~ form_diff_lastN + cum_pts_diff + cum_gd_diff,
  data = train_data_final,
  trace = FALSE)
proba_2526 <- as.matrix(predict(model_multinom, newdata = pred_2526, type = "probs"))
```

### (iii) Intégration des probabilités dans la base de données 
On ajoute les probabilités estimées directement dans la base pred_data.

```{r}
pred_2526_out <- pred_2526 %>%
  mutate(
    proba_H = proba_2526[, "H"],
    proba_D = proba_2526[, "D"],
    proba_A = proba_2526[, "A"],

    p_max = pmax(proba_H, proba_D, proba_A),

    pred_argmax = case_when(
      proba_H == p_max ~ "H",
      proba_D == p_max ~ "D",
      TRUE             ~ "A"
    )
  ) %>%
  arrange(Date)
```

### (iv) Illustration : matchs très déséquilibrés vs matchs très incertains 

Ici, on a les matchs les plus déséquilibrés :

```{r}
top10_pred_des <- pred_2526_out %>%
  arrange(desc(p_max)) %>%
  select(Date, HomeTeam, AwayTeam, proba_H, proba_D, proba_A, p_max) %>%
  head(10)
saveRDS(top10_pred_des, "top10_pred_des.rds")
knitr::kable(top10_pred_des)
```
Ces matchs oppposent généralement une équipe en forte dynamique à une équipe en difficulté.

Ici, on a les matchs les plus incertains :

```{r}
top10_pred_inc <- pred_2526_out %>%
arrange(p_max) %>%
select(Date, HomeTeam, AwayTeam, proba_H, proba_D, proba_A, p_max) %>%
head(10)
saveRDS(top10_pred_inc, "top10_pred_inc.rds")
knitr::kable(top10_pred_inc)
```
Ces matchs correspondent souvent à des équipes proches en termes de performance, des débuts de saison ou des confrontations historiquement équilibrées. 

## 2) Génération des probabilités de victoire pour les matchs futurs de la saison 25/26 

### (i) Ajout des rencontres non disponible dans la base de donnée 25/26 
Comme la base de donnée de la saison 25/26 contient que les matchs qui ont été déjà joué, il faut ajouter les matches restants de la saison 25/26. On peut maintenant avoir également une prédiction sur les matchs futurs.

```{r}
#Journées 17 à 22
fixtures_17_22 <- tribble(
  ~Date,         ~Round, ~HomeTeam,     ~AwayTeam,
  "2026-01-04",   17,     "Marseille",   "Nantes",
  "2026-01-04",   17,     "Brest",       "Auxerre",
  "2026-01-04",   17,     "Le Havre",    "Angers",
  "2026-01-04",   17,     "Lorient",     "Metz",
  "2026-01-04",   17,     "Paris SG",    "Paris FC",
  
  "2026-01-16",   18,     "Monaco",      "Lorient",
  "2026-01-16",   18,     "Paris SG",    "Lille",
  "2026-01-17",   18,     "Lens",        "Auxerre",
  "2026-01-17",   18,     "Toulouse",    "Nice",
  "2026-01-17",   18,     "Angers",      "Marseille",
  "2026-01-18",   18,     "Strasbourg",  "Metz",
  "2026-01-18",   18,     "Nantes",      "Paris FC",
  "2026-01-18",   18,     "Rennes",      "Le Havre",
  "2026-01-18",   18,     "Lyon",        "Brest",
  
  "2026-01-23",   19,     "Auxerre",     "Paris SG",
  "2026-01-24",   19,     "Rennes",      "Lorient",
  "2026-01-24",   19,     "Le Havre",    "Monaco",
  "2026-01-24",   19,     "Marseille",   "Lens",
  "2026-01-25",   19,     "Nantes",      "Nice",
  "2026-01-25",   19,     "Brest",       "Toulouse",
  "2026-01-25",   19,     "Metz",        "Lyon",
  "2026-01-25",   19,     "Paris FC",    "Angers",
  "2026-01-25",   19,     "Lille",       "Strasbourg",
  
  "2026-01-30",   20,     "Lens",        "Le Havre",
  "2026-01-31",   20,     "Paris FC",    "Marseille",
  "2026-01-31",   20,     "Lorient",     "Nantes",
  "2026-01-31",   20,     "Monaco",      "Rennes",
  "2026-02-01",   20,     "Lyon",        "Lille",
  "2026-02-01",   20,     "Angers",      "Metz",
  "2026-02-01",   20,     "Nice",        "Brest",
  "2026-02-01",   20,     "Toulouse",    "Auxerre",
  "2026-02-01",   20,     "Strasbourg",  "Paris SG",
  
  "2026-02-08",   21,     "Angers",      "Toulouse",
  "2026-02-08",   21,     "Auxerre",     "Paris FC",
  "2026-02-08",   21,     "Brest",       "Lorient",
  "2026-02-08",   21,     "Le Havre",    "Strasbourg",
  "2026-02-08",   21,     "Lens",        "Rennes",
  "2026-02-08",   21,     "Metz",        "Lille",
  "2026-02-08",   21,     "Nantes",      "Lyon",
  "2026-02-08",   21,     "Nice",        "Monaco",
  "2026-02-08",   21,     "Paris SG",    "Marseille",
  
  "2026-02-15",   22,     "Le Havre",    "Toulouse",
  "2026-02-15",   22,     "Lille",       "Brest",
  "2026-02-15",   22,     "Lorient",     "Angers",
  "2026-02-15",   22,     "Lyon",        "Nice",
  "2026-02-15",   22,     "Marseille",   "Strasbourg",
  "2026-02-15",   22,     "Metz",        "Auxerre",
  "2026-02-15",   22,     "Monaco",      "Nantes",
  "2026-02-15",   22,     "Paris FC",    "Lens",
  "2026-02-15",   22,     "Rennes",      "Paris SG")
```

```{r}
#Journées 23 à 32
fixtures_23_32 <- tribble(
  ~Round, ~Date,        ~HomeTeam,     ~AwayTeam,
  23, "2026-02-22", "Angers",      "Lille",
  23, "2026-02-22", "Auxerre",     "Rennes",
  23, "2026-02-22", "Brest",       "Marseille",
  23, "2026-02-22", "Lens",        "Monaco",
  23, "2026-02-22", "Nantes",      "Le Havre",
  23, "2026-02-22", "Nice",        "Lorient",
  23, "2026-02-22", "Paris SG",    "Metz",
  23, "2026-02-22", "Strasbourg",  "Lyon",
  23, "2026-02-22", "Toulouse",    "Paris FC",

  24, "2026-03-01", "Le Havre",    "Paris SG",
  24, "2026-03-01", "Lille",       "Nantes",
  24, "2026-03-01", "Lorient",     "Auxerre",
  24, "2026-03-01", "Marseille",   "Lyon",
  24, "2026-03-01", "Metz",        "Brest",
  24, "2026-03-01", "Monaco",      "Angers",
  24, "2026-03-01", "Paris FC",    "Nice",
  24, "2026-03-01", "Rennes",      "Toulouse",
  24, "2026-03-01", "Strasbourg",  "Lens",

  25, "2026-03-08", "Auxerre",     "Strasbourg",
  25, "2026-03-08", "Brest",       "Le Havre",
  25, "2026-03-08", "Lens",        "Metz",
  25, "2026-03-08", "Lille",       "Lorient",
  25, "2026-03-08", "Lyon",        "Paris FC",
  25, "2026-03-08", "Nantes",      "Angers",
  25, "2026-03-08", "Nice",        "Rennes",
  25, "2026-03-08", "Paris SG",    "Monaco",
  25, "2026-03-08", "Toulouse",    "Marseille",

  26, "2026-03-15", "Angers",      "Nice",
  26, "2026-03-15", "Le Havre",    "Lyon",
  26, "2026-03-15", "Lorient",     "Lens",
  26, "2026-03-15", "Marseille",   "Auxerre",
  26, "2026-03-15", "Metz",        "Toulouse",
  26, "2026-03-15", "Monaco",      "Brest",
  26, "2026-03-15", "Paris SG",    "Nantes",
  26, "2026-03-15", "Rennes",      "Lille",
  26, "2026-03-15", "Strasbourg",  "Paris FC",

  27, "2026-03-22", "Auxerre",     "Brest",
  27, "2026-03-22", "Lens",        "Angers",
  27, "2026-03-22", "Lyon",        "Monaco",
  27, "2026-03-22", "Marseille",   "Lille",
  27, "2026-03-22", "Nantes",      "Strasbourg",
  27, "2026-03-22", "Nice",        "Paris SG",
  27, "2026-03-22", "Paris FC",    "Le Havre",
  27, "2026-03-22", "Rennes",      "Metz",
  27, "2026-03-22", "Toulouse",    "Lorient",

  28, "2026-04-05", "Angers",      "Lyon",
  28, "2026-04-05", "Brest",       "Rennes",
  28, "2026-04-05", "Le Havre",    "Auxerre",
  28, "2026-04-05", "Lille",       "Lens",
  28, "2026-04-05", "Lorient",     "Paris FC",
  28, "2026-04-05", "Metz",        "Nantes",
  28, "2026-04-05", "Monaco",      "Marseille",
  28, "2026-04-05", "Paris SG",    "Toulouse",
  28, "2026-04-05", "Strasbourg",  "Nice",

  29, "2026-04-12", "Auxerre",     "Nantes",
  29, "2026-04-12", "Brest",       "Strasbourg",
  29, "2026-04-12", "Lens",        "Paris SG",
  29, "2026-04-12", "Lyon",        "Lorient",
  29, "2026-04-12", "Marseille",   "Metz",
  29, "2026-04-12", "Nice",        "Le Havre",
  29, "2026-04-12", "Paris FC",    "Monaco",
  29, "2026-04-12", "Rennes",      "Angers",
  29, "2026-04-12", "Toulouse",    "Lille",

  30, "2026-04-19", "Angers",      "Le Havre",
  30, "2026-04-19", "Lens",        "Toulouse",
  30, "2026-04-19", "Lille",       "Nice",
  30, "2026-04-19", "Lorient",     "Marseille",
  30, "2026-04-19", "Metz",        "Paris FC",
  30, "2026-04-19", "Monaco",      "Auxerre",
  30, "2026-04-19", "Nantes",      "Brest",
  30, "2026-04-19", "Paris SG",    "Lyon",
  30, "2026-04-19", "Strasbourg",  "Rennes",

  31, "2026-04-26", "Angers",      "Paris SG",
  31, "2026-04-26", "Brest",       "Lens",
  31, "2026-04-26", "Le Havre",    "Metz",
  31, "2026-04-26", "Lorient",     "Strasbourg",
  31, "2026-04-26", "Lyon",        "Auxerre",
  31, "2026-04-26", "Marseille",   "Nice",
  31, "2026-04-26", "Paris FC",    "Lille",
  31, "2026-04-26", "Rennes",      "Nantes",
  31, "2026-04-26", "Toulouse",    "Monaco",

  32, "2026-05-03", "Auxerre",     "Angers",
  32, "2026-05-03", "Lille",       "Le Havre",
  32, "2026-05-03", "Lyon",        "Rennes",
  32, "2026-05-03", "Metz",        "Monaco",
  32, "2026-05-03", "Nantes",      "Marseille",
  32, "2026-05-03", "Nice",        "Lens",
  32, "2026-05-03", "Paris FC",    "Brest",
  32, "2026-05-03", "Paris SG",    "Lorient",
  32, "2026-05-03", "Strasbourg",  "Toulouse")
```

```{r}
#Journées 32 à 34
fixtures_32_34 <- tribble(
  ~Round, ~Date,        ~HomeTeam,     ~AwayTeam,

  33, "2026-05-09", "Angers",      "Strasbourg",
  33, "2026-05-09", "Auxerre",     "Nice",
  33, "2026-05-09", "Le Havre",    "Marseille",
  33, "2026-05-09", "Lens",        "Nantes",
  33, "2026-05-09", "Metz",        "Lorient",
  33, "2026-05-09", "Monaco",      "Lille",
  33, "2026-05-09", "Paris SG",    "Brest",
  33, "2026-05-09", "Rennes",      "Paris FC",
  33, "2026-05-09", "Toulouse",    "Lyon",

  34, "2026-05-16", "Brest",       "Angers",
  34, "2026-05-16", "Lille",       "Auxerre",
  34, "2026-05-16", "Lorient",     "Le Havre",
  34, "2026-05-16", "Lyon",        "Lens",
  34, "2026-05-16", "Marseille",   "Rennes",
  34, "2026-05-16", "Nantes",      "Toulouse",
  34, "2026-05-16", "Nice",        "Metz",
  34, "2026-05-16", "Paris FC",    "Paris SG",
  34, "2026-05-16", "Strasbourg",  "Monaco")
```

```{r}
# Fonction de préparation des fixtures futures
prepare_fixtures <- function(df) {
  df %>%
    mutate(
      Date   = as.Date(Date),
      season = "2025-2026",
      FTHG   = NA_integer_,
      FTAG   = NA_integer_,
      FTR    = NA_character_)}

#Ajout des rencontres à notre base de donnée de la saison 25/26 en évitant les doublons
matches <- matches %>%
  bind_rows(
    list(fixtures_17_22, fixtures_23_32, fixtures_32_34) %>%
      map_dfr(prepare_fixtures) %>% # Appliquer la fonction à chaque fixture et empiler les data.frames obtenus ligne par ligne
      anti_join(matches, by = c("season", "Date", "HomeTeam", "AwayTeam")))
```

```{r}
#Vérification que les noms des équipes sont tous les mêmes
ref_teams <- sort(unique(c(matches$HomeTeam, matches$AwayTeam)))
hist_teams <- sort(unique(team_form$team))
unknown <- setdiff(ref_teams, hist_teams)
unknown
```

### (ii) Matchs futurs (post trêve)
Ici nous identifions les matchs futurs de la saison 2025–2026 joués après la trêve hivernale.
On filtre les rencontres sans résultat connu (FTR manquant), conserve uniquement les informations essentielles (date et équipes) et crée un identifiant unique de match (fixture_id).
La table obtenue constitue la base des matchs à prédire dans les étapes suivantes.

```{r}
date_treve <- as.Date("2026-01-01")
matches <- matches %>% mutate(Date = as.Date(Date))

future <- matches %>%
  filter(season == "2025-2026", is.na(FTR), Date >= date_treve) %>%
  transmute(
    season, Date, HomeTeam, AwayTeam,
    fixture_id = paste(season, Date, HomeTeam, AwayTeam, sep="|")
  ) %>%
  distinct(fixture_id, .keep_all = TRUE)
```

### (iii) Historique des matchs joués et transformation en équipe-match
Ici on extrait l’historique des matchs déjà joués et le transforme en une table équipe–match.
Pour chaque rencontre, on calcule les points obtenus et la différence de buts pour l’équipe à domicile et à l’extérieur, puis on passe au format long afin d’avoir une ligne par équipe et par match.
La table finale (team_games) constitue la base nécessaire au calcul des cumuls et de la forme récente des équipes.

```{r}
past <- matches %>%
  filter(!is.na(FTR)) %>%
  transmute(season, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR)

team_games <- past %>%
  mutate(
    home_pts = case_when(FTR=="H" ~ 3L, FTR=="D" ~ 1L, TRUE ~ 0L),
    away_pts = case_when(FTR=="A" ~ 3L, FTR=="D" ~ 1L, TRUE ~ 0L),
    home_gd  = FTHG - FTAG,
    away_gd  = FTAG - FTHG
  ) %>%
  pivot_longer(c(HomeTeam, AwayTeam), names_to="side", values_to="team") %>%
  mutate(
    points = if_else(side=="HomeTeam", home_pts, away_pts),
    gd     = if_else(side=="HomeTeam", home_gd,  away_gd)
  ) %>%
  select(season, Date, team, points, gd)
```

### (iv) Features par équipe avant chaque match
Ici on calcule pour chaque équipe et chaque saison, les features historiques disponibles avant chaque match.
Il construit les points cumulés et le différentiel de buts cumulés avant le match courant, puis la forme récente comme la moyenne des points sur les 5 derniers matchs, en excluant systématiquement le match du jour.
La table finale team_state synthétise l’état de forme et de performance de chaque équipe à chaque date.

```{r}
setDT(team_games)
setorder(team_games, season, team, Date)

team_games[, cum_pts_before := shift(cumsum(points), fill = 0), by=.(season, team)]
team_games[, cum_gd_before  := shift(cumsum(gd),     fill = 0), by=.(season, team)]

# forme = moyenne des N derniers points AVANT la date (donc on décale d'1)
team_games[, form_pts_lastN :=
             frollmean(shift(points, 1), n = N, align="right", fill=NA),
           by=.(season, team)]

# On garde seulement les colonnes utiles
team_state <- team_games[, .(season, Date, team, form_pts_lastN, cum_pts_before, cum_gd_before)]
```

### (v) Rolling join : dernier état connu avant chaque fixture
Ici nous rattachons à chaque match futur le dernier état connu de l’équipe à domicile et de l’équipe à l’extérieur avant la date du match.
On utilise donc un rolling join temporel (foverlaps) pour récupérer, par équipe et par saison, les features calculées lors du dernier match joué précédant chaque fixture.
Les vérifications finales garantissent qu’une seule ligne est associée à chaque match futur pour les équipes domicile et extérieure.

```{r}
setDT(future)

# intervalle de recherche pour foverlaps : [-inf, Date]
future[, start := as.Date("1900-01-01")]
future[, end   := Date]

team_state[, start := Date]
team_state[, end   := Date]

setkey(team_state, season, team, start, end)

home_q <- future[, .(fixture_id, season, Date, HomeTeam, AwayTeam, team=HomeTeam, start=as.Date("1900-01-01"), end=Date)]
away_q <- future[, .(fixture_id, season, Date, HomeTeam, AwayTeam, team=AwayTeam, start=as.Date("1900-01-01"), end=Date)]

setkey(home_q, season, team, start, end)
setkey(away_q, season, team, start, end)

home_state <- foverlaps(home_q, team_state, type="any", mult="last", nomatch=NA)
away_state <- foverlaps(away_q, team_state, type="any", mult="last", nomatch=NA)

# Vérif 1 ligne par fixture_id
stopifnot(home_state[, .N, by=fixture_id][, all(N==1)])
stopifnot(away_state[, .N, by=fixture_id][, all(N==1)])

```

### (vi) Construction table finale des features (future_feat) et différentiels des features
Ici on construit la table finale des features pour les matchs futurs en fusionnant, pour chaque fixture, l’état de l’équipe à domicile et de l’équipe à l’extérieur.
Puis on calcule ensuite les différentiels domicile – extérieur pour les trois variables retenues (forme récente, points cumulés, différence de buts cumulée).
Enfin, on conserve uniquement les matchs pour lesquels toutes les features sont disponibles et on vérifie qu’il existe bien des matchs prédictibles avant de poursuivre.

```{r}
future_feat <- merge(
  future[, .(fixture_id, season, Date, HomeTeam, AwayTeam)],
  home_state[, .(fixture_id,
                 home_form = form_pts_lastN,
                 home_cum_pts = cum_pts_before,
                 home_cum_gd  = cum_gd_before)],
  by="fixture_id", all.x=TRUE
)

future_feat <- merge(
  future_feat,
  away_state[, .(fixture_id,
                 away_form = form_pts_lastN,
                 away_cum_pts = cum_pts_before,
                 away_cum_gd  = cum_gd_before)],
  by="fixture_id", all.x=TRUE
)

future_feat[, form_diff_lastN := home_form    - away_form]
future_feat[, cum_pts_diff    := home_cum_pts - away_cum_pts]
future_feat[, cum_gd_diff     := home_cum_gd  - away_cum_gd]

# On garde seulement les matchs où les 3 features sont dispo
to_predict_ok <- future_feat[!is.na(form_diff_lastN) & !is.na(cum_pts_diff) & !is.na(cum_gd_diff)]

cat("Matchs futurs trouvés :", nrow(future_feat), "\n")
cat("Matchs prédictibles (features OK) :", nrow(to_predict_ok), "\n")

if (nrow(to_predict_ok) == 0) {
  stop("0 match prédictible : il manque de l'historique pour au moins une équipe (forme/cumul NA).")
}
```
Matchs futurs trouvés : 153 la date post-trêve et la présence de fixtures uniques a identifié 153 matchs à venir.

Matchs prédictibles (features OK) : 153 veut dire que pour tous ces 153 matchs, nous avons bien les trois features (form_diff_lastN, cum_pts_diff, cum_gd_diff) calculées, donc il n’y a pas de données manquantes empêchant la prédiction.

### (vii) Prédiction (sécuriser H/D/A)
Ici on applique le modèle multinomial entraîné pour prédire les résultats H/D/A des matchs futurs pour lesquels les features sont disponibles. On sécurise le résultat pour gérer les cas particuliers :
- si un seul match, on force la sortie en matrice 1x3
- si certaines colonnes H/D/A sont manquantes, on les crée avec des 0
Les colonnes sont ensuite réordonnées H / D / A pour cohérence.
Résultat : `proba_future` contient les probabilités de chaque issue pour chaque match.

```{r}
proba_future <- predict(model_multinom, newdata = as.data.frame(to_predict_ok), type = "probs")
proba_future <- as.matrix(proba_future)

# Si predict renvoie un vecteur (1 seule ligne), on force en matrice 1xK
if (is.null(dim(proba_future))) {
  proba_future <- matrix(proba_future, nrow=1, dimnames=list(NULL, names(proba_future)))
}

# Assurer colonnes H/D/A
for (lvl in c("H","D","A")) {
  if (!(lvl %in% colnames(proba_future))) {
    proba_future <- cbind(proba_future, setNames(rep(0, nrow(proba_future)), lvl))
  }
}
proba_future <- proba_future[, c("H","D","A"), drop=FALSE]
```

### (viii) Table finale des prédictions
Finalement ici  nous construisons la table finale des prédictions pour les matchs futurs.
Étapes principales :
- On combine `to_predict_ok` et les probabilités prédites `proba_future`.
- Création des colonnes `proba_H`, `proba_D`, `proba_A` pour chaque issue.
- Calcul de `p_max`, la probabilité la plus élevée parmi H/D/A.
- Détermination d'une prédiction initiale `pred_argmax` par argmax.
- Tri des matchs par date et équipes, puis sélection des colonnes principales.
- Affichage des 50 premiers matchs pour vérification.
- Création d'une colonne sécurisée `pred_class` pour la classe prédite finale.
Résultat : `fiche_predictions` contient pour chaque match futur les probabilités et la prédiction finale (H/D/A) de manière ordonnée.

```{r}
fiche_predictions <- as.data.frame(to_predict_ok) %>%
  mutate(
    proba_H = proba_future[, "H"],
    proba_D = proba_future[, "D"],
    proba_A = proba_future[, "A"],
    p_max   = pmax(proba_H, proba_D, proba_A),
    pred_argmax = case_when(
      proba_H == p_max ~ "H",
      proba_D == p_max ~ "D",
      TRUE             ~ "A"
    ))%>%
  arrange(Date, HomeTeam, AwayTeam) %>%
  select(Date, HomeTeam, AwayTeam, proba_H, proba_D, proba_A, p_max)

print(head(fiche_predictions, n = 50))
fiche_predictions <- fiche_predictions %>%
  mutate(
    pred_class = case_when(
      p_max == proba_H ~ "H",
      p_max == proba_D ~ "D",
      p_max == proba_A ~ "A",
      TRUE ~ NA_character_  # Sécurité si une valeur est manquante
      ))
```

### (xix) L'interface

```{r}
ui <- fluidPage(
  
  theme = shinytheme("flatly"),
  
  titlePanel("Prédiction des probabilités de match"),
  
  sidebarLayout(
    
    sidebarPanel(
      width = 3,
      
      h4("Sélection du match"),
      
      selectInput(
        inputId = "match",
        label = "Match",
        choices = NULL,
        selectize = TRUE
      ),
      
      hr(),
      
      helpText(
        "Tape le nom d’une équipe, une date ou fais défiler la liste."
      )
    ),
    
    mainPanel(
      
      fluidRow(
        
        column(
          6,
          h3(textOutput("match_title")),
          h5(textOutput("match_date"))
        ),
        
        column(
          6,
          h3("Issue la plus probable"),
          h4(
            textOutput("predicted_outcome"),
            style = "color:#2C3E50; font-weight:bold;"
          )
        )
      ),
      
      hr(),
      
      plotOutput("proba_plot", height = "350px")
    )
  )
)
```

```{r}
server <- function(input, output, session) {
  
  # Création d'un label lisible pour l'utilisateur
  matches_ui <- fiche_predictions %>%
    mutate(
      match_label = paste0(
        format(Date, "%d/%m/%Y"),
        " – ",
        HomeTeam, " vs ", AwayTeam
      )
    )
  
  # Initialisation du selectInput avec options premium
 updateSelectizeInput(
  session,
  "match",
  choices = matches_ui$match_label,
  selected = matches_ui$match_label[1],
  options = list(
    placeholder = "Ex: Paris, Marseille, 17/08...",
    allowClear = TRUE,
    maxOptions = 5000
  )
)

  
  # Match sélectionné
  selected_match <- reactive({
    matches_ui %>%
      filter(match_label == input$match)
  })
  
  # Titre du match
  output$match_title <- renderText({
    paste(
      selected_match()$HomeTeam,
      "vs",
      selected_match()$AwayTeam
    )
  })
  
  # Date du match
  output$match_date <- renderText({
    paste(
      "Date :",
      format(selected_match()$Date, "%d %B %Y")
    )
  })
  
  # Issue la plus probable 
  output$predicted_outcome <- renderText({
    dplyr::recode(
      selected_match()$pred_class,
      "H" = "Victoire domicile",
      "D" = "Match nul",
      "A" = "Victoire extérieur"
    )
  })
  
  # Graphique des probabilités
  output$proba_plot <- renderPlot({
    
    df_plot <- tibble(
      Issue = c("Victoire domicile", "Match nul", "Victoire extérieur"),
      Probabilité = c(
        selected_match()$proba_H,
        selected_match()$proba_D,
        selected_match()$proba_A
      )
    )
    
    ggplot(df_plot, aes(x = Issue, y = Probabilité, fill = Issue)) +
      geom_col(width = 0.6, show.legend = FALSE) +
      geom_text(
        aes(label = scales::percent(Probabilité, accuracy = 0.1)),
        vjust = -0.5,
        size = 5
      ) +
      scale_y_continuous(
        labels = scales::percent,
        limits = c(0, 1)
      ) +
      scale_fill_manual(
        values = c("#2ECC71", "#F1C40F", "#E74C3C")
      ) +
      labs(
        y = "Probabilité",
        x = NULL
      ) +
      theme_minimal(base_size = 14)
  })
}
```

```{r}
shinyApp(ui, server)
```
