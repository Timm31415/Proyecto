# =========================================================
# PROYECTO FINAL - CIENCIA DE DATOS
# REGRESIÓN LINEAL MÚLTIPLE - AIRBNB
# Alumno: Luis Guillermo Timm  Sosa
# Matrícula: AL07117001@Tecmilenio.mx
# =========================================================

# =========================
# LIBRERÍAS
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# =========================
# CARGAR DATASET
# =========================


df = pd.read_csv("Airbnb_Al07117001.csv")

# Mostrar primeras filas
print("\nDATASET")
print(df.head())

# =========================
# VARIABLES
# =========================

# Variables independientes
X = df[["accommodates", "bedrooms"]]

# Variable dependiente
y = df["price_usd"]

# =========================
# MODELO DE REGRESIÓN
# =========================

modelo = LinearRegression()

# Entrenar modelo
modelo.fit(X, y)

# =========================
# ECUACIÓN DEL MODELO
# =========================

intercepto = modelo.intercept_
coeficientes = modelo.coef_

print("\nECUACIÓN DEL MODELO")
print(f"Intercepto: {intercepto}")
print(f"Coeficiente accommodates: {coeficientes[0]}")
print(f"Coeficiente bedrooms: {coeficientes[1]}")

# Ecuación
print("\nEcuación:")
print(f"Y = {intercepto:.2f} + ({coeficientes[0]:.2f} * accommodates) + ({coeficientes[1]:.2f} * bedrooms)")

# =========================================================
# PASO 9 - PREDICCIONES
# =========================================================

# Predicciones de todos los datos
y_pred = modelo.predict(X)


df["predicciones"] = y_pred

print("\nPREDICCIONES")
print(df[["accommodates", "bedrooms", "price_usd", "predicciones"]])

# Nuevas predicciones
nuevos_datos = pd.DataFrame({
    "accommodates": [2, 4, 6],
    "bedrooms": [1, 2, 3]
})

predicciones_nuevas = modelo.predict(nuevos_datos)

print("\nNUEVAS PREDICCIONES")

for i in range(len(nuevos_datos)):
    print(
        f"accommodates = {nuevos_datos['accommodates'][i]}, "
        f"bedrooms = {nuevos_datos['bedrooms'][i]} "
        f"=> Predicción = {predicciones_nuevas[i]:.2f}"
    )

# =========================================================
# PASO 11 - RESIDUALES
# =========================================================

# Calcular residuales
residuales = y - y_pred

# Agregar al dataframe
df["residuales"] = residuales

print("\nRESIDUALES")
print(df[["price_usd", "predicciones", "residuales"]])

# Residual mínimo y máximo
print("\nResidual mínimo:", residuales.min())
print("Residual máximo:", residuales.max())

# =========================================================
# PASO 12 - DESVIACIÓN ESTÁNDAR
# =========================================================

desviacion = np.std(residuales)

print("\nDESVIACIÓN ESTÁNDAR DE RESIDUALES")
print(desviacion)

# =========================================================
# PASO 13 - CORRELACIÓN DE PEARSON
# =========================================================

corr_x1_y, _ = pearsonr(df["accommodates"], y)
corr_x2_y, _ = pearsonr(df["bedrooms"], y)
corr_x1_x2, _ = pearsonr(df["accommodates"], df["bedrooms"])

print("\nCORRELACIONES")

print(f"Correlación accommodates y price_usd: {corr_x1_y}")
print(f"Correlación bedrooms y price_usd: {corr_x2_y}")
print(f"Correlación accommodates y bedrooms: {corr_x1_x2}")

# =========================================================
# PASO 14 - R²
# =========================================================

r2 = r2_score(y, y_pred)

print("\nR²")
print(r2)

print(f"Porcentaje explicado: {r2*100:.2f}%")

# =========================================================
# PASO 15 - RMSE
# =========================================================

rmse = np.sqrt(mean_squared_error(y, y_pred))

print("\nRMSE")
print(rmse)

# =========================================================
# PASOS 16 Y 17 - REPORTE OLS
# =========================================================

# Agregar constante
X_sm = sm.add_constant(X)

# Modelo OLS
modelo_ols = sm.OLS(y, X_sm).fit()

print("\nREPORTE OLS")
print(modelo_ols.summary())

# =========================================================
# PASO 18 - RESIDUOS VS PREDICHOS
# =========================================================

# Se hará en el dashboard

# =========================================================
# PASO 19 - DURBIN WATSON
# =========================================================

dw = durbin_watson(residuales)

print("\nDURBIN-WATSON")
print(dw)

# =========================================================
# PASO 20 - BREUSCH PAGAN
# =========================================================

bp_test = het_breuschpagan(residuales, X_sm)

bp_pvalue = bp_test[1]

print("\nBREUSCH-PAGAN")
print("P-value:", bp_pvalue)

# =========================================================
# PASO 21 - QQ PLOT
# =========================================================

# Se hará en el dashboard

# =========================================================
# PASO 10 - DASHBOARD FINAL
# =========================================================

# Crear ventana con 4 gráficas
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# ---------------------------------------------------------
# GRÁFICA 1
# accommodates vs price_usd
# ---------------------------------------------------------

axs[0, 0].scatter(df["accommodates"], y)
axs[0, 0].set_title("accommodates vs price_usd")
axs[0, 0].set_xlabel("accommodates")
axs[0, 0].set_ylabel("price_usd")

# ---------------------------------------------------------
# GRÁFICA 2
# bedrooms vs price_usd
# ---------------------------------------------------------

axs[0, 1].scatter(df["bedrooms"], y)
axs[0, 1].set_title("bedrooms vs price_usd")
axs[0, 1].set_xlabel("bedrooms")
axs[0, 1].set_ylabel("price_usd")

# ---------------------------------------------------------
# GRÁFICA 3
# residuos vs predichos
# ---------------------------------------------------------

axs[1, 0].scatter(y_pred, residuales)
axs[1, 0].axhline(y=0)
axs[1, 0].set_title("Residuos vs Predichos")
axs[1, 0].set_xlabel("Predichos")
axs[1, 0].set_ylabel("Residuales")

# ---------------------------------------------------------
# GRÁFICA 4
# QQ-PLOT
# ---------------------------------------------------------

sm.qqplot(residuales, line='45', ax=axs[1, 1])

axs[1, 1].set_title("QQ Plot")

# ---------------------------------------------------------
# AJUSTAR ESPACIOS
# ---------------------------------------------------------

plt.tight_layout()

# Guardar dashboard
plt.savefig("dashboard_TuMatricula.png")

# Mostrar dashboard
plt.show()

# =========================================================
# GUARDAR NUEVO DATASET
# =========================================================

df.to_csv("Airbnb_Limpio_TuMatricula.csv", index=False)

print("\nDATASET GUARDADO")

# =========================================================
# FIN DEL PROYECTO
# =========================================================
