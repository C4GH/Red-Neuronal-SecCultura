# Cargar librerías necesarias
library(dplyr)
library(readr)

# Cargar datos
setwd("~/Acatlán/Titulación/Datos")  # Ajustar al directorio adecuado
datos_modelo <- read_csv("DatosIA.csv", locale = locale(encoding = "UTF-8"))
datos_usuarios <- read_csv("test_ia_202405201501.csv", locale = locale(encoding = "UTF-8"))

# Preparar datos de usuario y combinar
datos_usuarios_agrupados <- datos_usuarios %>%
  group_by(pregunta) %>%
  summarise(
    prob1_x = sum(respuesta == 1) / n(),
    prob2_x = sum(respuesta == 2) / n(),
    prob3_x = sum(respuesta == 3) / n(),
    .groups = 'drop'
  )

datos_combinados <- full_join(datos_usuarios_agrupados, datos_modelo, by = "pregunta")

# Definir la función para calcular la divergencia de Kullback-Leibler
calculate_kl_divergence <- function(user_probs, model_probs) {
  # Normalizar las probabilidades
  user_probs <- user_probs / sum(user_probs)
  model_probs <- model_probs / sum(model_probs)
  
  # Reemplazar ceros para evitar logaritmos de cero
  user_probs[user_probs == 0] <- 1e-10
  model_probs[model_probs == 0] <- 1e-10
  
  # Calcular la divergencia de KL
  kl_divergence <- sum(user_probs * log(user_probs / model_probs))
  return(kl_divergence)
}

# Aplicar la función para calcular la divergencia de KL
results <- data.frame(pregunta = datos_combinados$pregunta, 
                      cadena = datos_combinados$cadena,  # Incluir la cadena de la pregunta
                      kl_divergence = NA_real_)
for (i in seq_len(nrow(datos_combinados))) {
  user_probs <- as.numeric(datos_combinados[i, c("prob1_x", "prob2_x", "prob3_x")])
  model_probs <- as.numeric(datos_combinados[i, c("prob1", "prob2", "prob3")])
  
  results$kl_divergence[i] <- calculate_kl_divergence(user_probs, model_probs)
}

# Ver los resultados
print(results)

# Opcional: Guardar los resultados en un archivo CSV
write.csv(results, "KL_Divergence_Results.csv", row.names = FALSE)




