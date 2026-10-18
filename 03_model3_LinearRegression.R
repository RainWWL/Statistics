# =============================================================================
# TIME SERIES PROJECT — MODEL 3: LINEAR REGRESSION (tslm)
# Member responsible: Member 3
# Reference: Practical 6 (Linear Model + Decomposition)
# =============================================================================

library(forecast)
library(tseries)
library(lmtest)   # for dwtest (Durbin-Watson)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
source("00_data_prep.R")
prep <- prepare_data("Sales.csv")

# 1. Core time series objects
train_clean <- prep$train_clean
test_ts  <- prep$test_ts

# 2. Variables specifically needed for your Linear Regression script
n_train  <- prep$n_train  # You use this to calculate the 'Trend' variable in test_df_3b
n_test   <- prep$n_test   # You use this for the test loop and 'Trend' sequence
data     <- prep$data     # You use this in Section 7 to get the Date column (data$Month) for your CSV

cat("=== Linear Model (tslm) ===\n")
cat("Training observations:", length(train_clean), "\n")
cat("Test observations    :", length(test_ts), "\n\n")

# ── Helper: compute full accuracy table from vectors ─────────────────────────
# Produces ME, RMSE, MAE, MPE, MAPE, MASE, ACF1, Theil's U
# for both Training set and Test set rows.
calc_accuracy <- function(actual_tr, fitted_tr, actual_te, pred_te) {
  
  calc_row <- function(actual, predicted, naive_errors) {
    e          <- actual - predicted
    pe         <- e / actual * 100
    acf1_val   <- if (length(e) > 2) acf(e, lag.max = 1, plot = FALSE)$acf[2] else NA
    mae_naive  <- mean(abs(naive_errors))
    theils_u   <- sqrt(mean(e^2)) / sqrt(mean(naive_errors^2))
    data.frame(
      ME      = mean(e),
      RMSE    = sqrt(mean(e^2)),
      MAE     = mean(abs(e)),
      MPE     = mean(pe),
      MAPE    = mean(abs(pe)),
      MASE    = mean(abs(e)) / mae_naive,
      ACF1    = acf1_val,
      Theil.U = theils_u
    )
  }
  
  # Training: naive benchmark = random walk (one-step-ahead)
  tr_naive <- diff(actual_tr)                            # length n-1
  row_tr   <- calc_row(actual_tr[-1], fitted_tr[-1], tr_naive)
  
  # Test: naive benchmark = last training value as flat forecast
  te_naive <- actual_te - c(tail(actual_tr, 1), head(actual_te, -1))
  row_te   <- calc_row(actual_te, pred_te, te_naive)
  
  out            <- rbind(row_tr, row_te)
  rownames(out)  <- c("Training set", "Test set")
  out
}

# ── 2. MODEL IDENTIFICATION ───────────────────────────────────────────────────
plot(ts_data, main = "Sales Time Series", ylab = "Sales")

components <- decompose(train_clean, type = "multiplicative")
plot(components)
checkresiduals(components$random)

# =============================================================================
# SECTION 2.5 — STATIONARITY VERIFICATION
# WHY: tslm() models non-stationarity explicitly via Trend + Season predictors,
#      but we verify the raw series here for rigour and transparency.
#
# Tests performed:
#   ADF test  : H0 = non-stationary | p < 0.05 -> stationary
#   KPSS test : H0 = stationary     | p < 0.05 -> non-stationary
#   ndiffs()  : recommended number of regular differences needed
#   nsdiffs() : recommended number of seasonal differences needed
# =============================================================================

cat("=== SECTION 2.5: Stationarity Verification ===\n\n")

# --- Raw series ---
cat("--- Raw Training Series ---\n")
cat("H0 (ADF) : series is non-stationary | p < 0.05 = stationary\n")
adf_raw <- adf.test(train_clean)
print(adf_raw)
cat("H0 (KPSS): series is stationary     | p < 0.05 = non-stationary\n")
kpss_raw <- kpss.test(train_clean)
print(kpss_raw)

ndiffs_val  <- ndiffs(train_clean)
nsdiffs_val <- nsdiffs(train_clean)
cat("Suggested regular differences  (d) :", ndiffs_val,  "\n")
cat("Suggested seasonal differences (D) :", nsdiffs_val, "\n\n")

# --- ACF/PACF of raw series ---
layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
plot(train_clean, ylab = "Sales", main = "Sales Training Series (Raw)")
acf(train_clean,  lag.max = 40, main = "ACF  — Raw Series")
pacf(train_clean, lag.max = 40, main = "PACF — Raw Series")
layout(matrix(1))

# --- After seasonal differencing (lag=12) ---
SDiff_lm <- ts(diff(train_clean, lag = 12), frequency = 12)
cat("--- After Seasonal Differencing (lag=12) ---\n")
adf_sdiff <- adf.test(SDiff_lm)
print(adf_sdiff)
layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
plot.ts(SDiff_lm, main = "After Seasonal Differencing (lag=12)", ylab = "Sales")
acf(SDiff_lm,  lag.max = 40, main = "ACF  — Seasonal Diff")
pacf(SDiff_lm, lag.max = 40, main = "PACF — Seasonal Diff")
layout(matrix(1))

# --- After regular differencing (lag=1) on top of seasonal ---
Diff_lm <- ts(diff(SDiff_lm, lag = 1), frequency = 12)
cat("--- After Regular Differencing (lag=1) ---\n")
adf_diff <- adf.test(Diff_lm)
print(adf_diff)
layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
plot.ts(Diff_lm, main = "After Regular Differencing (lag=1)", ylab = "Sales")
acf(Diff_lm,  lag.max = 40, main = "ACF  — Nonseasonal Diff")
pacf(Diff_lm, lag.max = 40, main = "PACF — Nonseasonal Diff")
layout(matrix(1))

# --- Summary ---
cat("=== Stationarity Summary ===\n")
cat(sprintf("Raw series      — ADF p=%.4f (%s) | KPSS p=%.4f (%s)\n",
            adf_raw$p.value,
            ifelse(adf_raw$p.value  < 0.05, "stationary",     "non-stationary"),
            kpss_raw$p.value,
            ifelse(kpss_raw$p.value < 0.05, "non-stationary", "stationary")))
cat(sprintf("After SDiff     — ADF p=%.4f (%s)\n",
            adf_sdiff$p.value,
            ifelse(adf_sdiff$p.value < 0.05, "stationary", "non-stationary")))
cat(sprintf("After SDiff+Diff— ADF p=%.4f (%s)\n",
            adf_diff$p.value,
            ifelse(adf_diff$p.value  < 0.05, "stationary", "non-stationary")))
cat("\nNOTE: tslm() addresses non-stationarity via explicit Trend + Season terms.\n")
cat("The differencing above is for verification only — train_clean is passed\n")
cat("to tslm() unchanged.\n\n")

cat("=== STL + Random Walk (shortcut, Practical 6) ===\n")
fcast_stl <- stlf(train_clean, method = "naive", length(test_ts))
summary(fcast_stl)
checkresiduals(fcast_stl)
plot(fcast_stl)
acc_stl <- accuracy(fcast_stl, test_ts)

# ── 3. MODEL FITTING ──────────────────────────────────────────────────────────

# --- 3a. Trend-only -----------------------------------------------------------
cat("\n=== Model 3a: Trend Only — tslm(train ~ trend) ===\n")
fit_trend <- tslm(train_clean ~ trend)
summary(fit_trend)
checkresiduals(fit_trend$residuals)
fc_trend <- forecast(fit_trend, h = n_test)
plot(train_clean,
     main = "Trend-Only Linear Model Forecast",
     ylab = "Sales", xlab = "Time",
     xlim = c(2015, 2021),
     ylim = range(c(train_clean, fc_trend$mean, fc_trend$lower, fc_trend$upper, test_ts), na.rm = TRUE))
lines(fc_trend$mean,        col = "blue")
lines(fc_trend$lower[, 2],  col = "gray", lty = 2)
lines(fc_trend$upper[, 2],  col = "gray", lty = 2)
lines(test_ts, col = "turquoise2")
legend("topleft", legend = c("Training", "Forecast", "Actual", "95% CI"),
       col = c("black", "blue", "turquoise2", "gray"), lty = c(1, 1, 1, 2))

acc_3a <- calc_accuracy(
  actual_tr = as.numeric(train_clean),
  fitted_tr = as.numeric(fitted(fit_trend)),
  actual_te = as.numeric(test_ts),
  pred_te   = as.numeric(fc_trend$mean)
)
cat("\n=== Model 3a Accuracy ===\n")
print(round(acc_3a, 4))

# Diagnostic tests for 3a
dw_3a     <- dwtest(fit_trend)
lb_3a     <- Box.test(residuals(fit_trend), lag = 20, type = "Ljung-Box")
sw_3a     <- shapiro.test(residuals(fit_trend))
r2_3a     <- summary(fit_trend)$r.squared
adj_r2_3a <- summary(fit_trend)$adj.r.squared

# --- 3b. Trend + Season + Lag5 -----------------------------------------------
cat("\n=== Model 3b: Trend + Season + Lag5 ===\n")

lag5_train      <- stats::lag(train_clean, -5)
train_intersect <- ts.intersect(train_clean, lag5_train)

train_clean_al  <- as.numeric(train_intersect[, "train_clean"])
lag5_train_al   <- as.numeric(train_intersect[, "lag5_train"])
n_aligned       <- length(train_clean_al)

train_df_3b <- data.frame(
  Sales  = train_clean_al,
  Trend  = 6:(5 + n_aligned),
  Season = factor(cycle(train_intersect[, "train_clean"]), levels = 1:12),
  Lag5   = lag5_train_al
)

fit_ts <- lm(Sales ~ Trend + Season + Lag5, data = train_df_3b)
summary(fit_ts)
checkresiduals(residuals(fit_ts))

# Outlier check
cat("\nOutlier Check (residuals > 2 SD):\n")
res_sd      <- sd(residuals(fit_ts))
outlier_idx <- which(abs(residuals(fit_ts)) > 2 * res_sd)
if (length(outlier_idx) > 0) {
  cat("Potential outlier observations at index:", outlier_idx, "\n")
  cat("Residual values:", round(residuals(fit_ts)[outlier_idx], 2), "\n")
} else {
  cat("No outliers detected beyond 2 SD.\n")
}

# Build test data frame
full_ts_vec    <- c(as.numeric(train_clean), as.numeric(test_ts))
lag5_test_vals <- full_ts_vec[(n_train - 4):(n_train - 4 + n_test - 1)]

test_df_3b <- data.frame(
  Trend  = (n_train + 1):(n_train + n_test),
  Season = factor(cycle(test_ts), levels = 1:12),
  Lag5   = lag5_test_vals
)

pred_3b     <- predict(fit_ts, newdata = test_df_3b, interval = "prediction")
fc_ts_mean  <- ts(pred_3b[, "fit"], start = start(test_ts), frequency = 12)
fc_ts_lower <- ts(pred_3b[, "lwr"], start = start(test_ts), frequency = 12)
fc_ts_upper <- ts(pred_3b[, "upr"], start = start(test_ts), frequency = 12)

plot(train_clean,
     main = "Trend + Season + Lag5 (3b) Forecast",
     ylab = "Sales", xlab = "Time",
     xlim = c(2015, 2021),
     ylim = range(c(train_clean, fc_ts_lower, fc_ts_upper, test_ts), na.rm = TRUE))
lines(fc_ts_mean,  col = "blue")
lines(fc_ts_lower, col = "gray", lty = 2)
lines(fc_ts_upper, col = "gray", lty = 2)
lines(test_ts,     col = "turquoise2")
legend("topleft", legend = c("Training", "Forecast", "Actual", "95% CI"),
       col = c("black", "blue", "turquoise2", "gray"), lty = c(1, 1, 1, 2))

acc_3b <- calc_accuracy(
  actual_tr = train_df_3b$Sales,
  fitted_tr = as.numeric(fitted(fit_ts)),
  actual_te = as.numeric(test_ts),
  pred_te   = as.numeric(pred_3b[, "fit"])
)
cat("\n=== Model 3b Accuracy ===\n")
print(round(acc_3b, 4))



# ── 4. MODEL DIAGNOSTICS (3b) ─────────────────────────────────────────────────
cat("\n=== Model Diagnostics: Trend + Season + Lag5 Model (3b) ===\n")

par(mfrow = c(2, 2))
plot(fit_ts)
par(mfrow = c(1, 1))

layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
plot(residuals(fit_ts), main = "Residuals Over Time", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)
acf(residuals(fit_ts),  lag.max = 36, main = "ACF of Residuals")
pacf(residuals(fit_ts), lag.max = 36, main = "PACF of Residuals")
layout(matrix(1))

cat("\nDurbin-Watson Test for Autocorrelation:\n")
dw_3b <- dwtest(fit_ts)
print(dw_3b)
cat("DW≈2 → no autocorrelation; DW<1.5 → positive autocorrelation\n\n")

lb_3b <- Box.test(residuals(fit_ts), lag = 20, type = "Ljung-Box")
cat("Ljung-Box Test:\n"); print(lb_3b)

sw_3b <- shapiro.test(residuals(fit_ts))
cat("\nShapiro-Wilk Test:\n"); print(sw_3b)

r2_3b     <- summary(fit_ts)$r.squared
adj_r2_3b <- summary(fit_ts)$adj.r.squared

# ── 5. FORECAST (best model: 3b) ──────────────────────────────────────────────
cat("\n=== Forecast: Trend + Season + Lag5 Model (3b) ===\n")
plot(train_clean,
     main = "Linear Model (Trend + Season + Lag5) Forecast",
     ylab = "Sales", xlab = "Time",
     xlim = c(2015, 2021),
     ylim = range(c(train_clean, fc_ts_lower, fc_ts_upper, test_ts), na.rm = TRUE))
lines(fc_ts_mean,  col = "blue")
lines(fc_ts_lower, col = "gray", lty = 2)
lines(fc_ts_upper, col = "gray", lty = 2)
lines(test_ts,     col = "turquoise2")
legend("topleft", legend = c("Training", "Forecast", "Actual", "95% CI"),
       col = c("black", "blue", "turquoise2", "gray"), lty = c(1, 1, 1, 2))

# ── 6. ACCURACY METRICS — ALL MODELS COMPARED ────────────────────────────────
cat("\n================================================================\n")
cat("=== Full Accuracy Comparison: All Models ===\n")
cat("================================================================\n")

# --- Training Set Metrics ---
cat("\n--- Training Set Metrics ---\n")
train_comparison <- data.frame(
  Model = c("STL + Random Walk", "3a: Trend Only", "3b: Trend+Season+Lag5"),
  ME    = c(acc_stl["Training set","ME"],   acc_3a["Training set","ME"],   acc_3b["Training set","ME"]),
  RMSE  = c(acc_stl["Training set","RMSE"], acc_3a["Training set","RMSE"], acc_3b["Training set","RMSE"]),
  MAE   = c(acc_stl["Training set","MAE"],  acc_3a["Training set","MAE"],  acc_3b["Training set","MAE"]),
  MPE   = c(acc_stl["Training set","MPE"],  acc_3a["Training set","MPE"],  acc_3b["Training set","MPE"]),
  MAPE  = c(acc_stl["Training set","MAPE"], acc_3a["Training set","MAPE"], acc_3b["Training set","MAPE"]),
  MASE  = c(acc_stl["Training set","MASE"], acc_3a["Training set","MASE"], acc_3b["Training set","MASE"]),
  ACF1  = c(acc_stl["Training set","ACF1"], acc_3a["Training set","ACF1"], acc_3b["Training set","ACF1"])
)
{
  tmp <- train_comparison
  tmp[-1] <- round(tmp[-1], 4)
  print(tmp)
}

# --- Test Set Metrics ---
cat("\n--- Test Set Metrics ---\n")
test_comparison <- data.frame(
  Model   = c("STL + Random Walk", "3a: Trend Only", "3b: Trend+Season+Lag5"),
  ME      = c(acc_stl["Test set","ME"],   acc_3a["Test set","ME"],   acc_3b["Test set","ME"]),
  RMSE    = c(acc_stl["Test set","RMSE"], acc_3a["Test set","RMSE"], acc_3b["Test set","RMSE"]),
  MAE     = c(acc_stl["Test set","MAE"],  acc_3a["Test set","MAE"],  acc_3b["Test set","MAE"]),
  MPE     = c(acc_stl["Test set","MPE"],  acc_3a["Test set","MPE"],  acc_3b["Test set","MPE"]),
  MAPE    = c(acc_stl["Test set","MAPE"], acc_3a["Test set","MAPE"], acc_3b["Test set","MAPE"]),
  MASE    = c(acc_stl["Test set","MASE"], acc_3a["Test set","MASE"], acc_3b["Test set","MASE"]),
  ACF1    = c(acc_stl["Test set","ACF1"], acc_3a["Test set","ACF1"], acc_3b["Test set","ACF1"]),
  Theil.U = c(NA,                         acc_3a["Test set","Theil.U"], acc_3b["Test set","Theil.U"])
)
{
  tmp <- test_comparison
  tmp[-1] <- round(tmp[-1], 4)
  print(tmp)
}

# Best model per metric (lower = better, based on Test set)
cat("\n=== Best Model Per Metric (Test Set) ===\n")
for (m in c("RMSE","MAE","MAPE","MASE","Theil.U")) {
  vals     <- as.numeric(test_comparison[[m]])
  best_idx <- which.min(vals)
  cat(sprintf("%-10s → %s (%.4f)\n", m, test_comparison$Model[best_idx], vals[best_idx]))
}

best_overall <- test_comparison$Model[which.min(test_comparison$MAPE)]
cat("\n>>> Best overall model by Test MAPE:", best_overall, "<<<\n")

# Keep combined comparison for CSV export
comparison <- test_comparison

# ── 6b. DIAGNOSTIC TESTS COMPARISON ──────────────────────────────────────────
cat("\n================================================================\n")
cat("=== Diagnostic Tests Comparison: All Models ===\n")
cat("================================================================\n")

diag_comparison <- data.frame(
  Model     = c("STL + Random Walk", "3a: Trend Only", "3b: Trend+Season+Lag5"),
  DW        = c(NA,                        round(dw_3a$statistic,  4), round(dw_3b$statistic,  4)),
  DW_pval   = c(NA,                        round(dw_3a$p.value,    4), round(dw_3b$p.value,    4)),
  LB_pval   = c(NA,                        round(lb_3a$p.value,    4), round(lb_3b$p.value,    4)),
  SW_pval   = c(NA,                        round(sw_3a$p.value,    4), round(sw_3b$p.value,    4)),
  R2        = c(NA,                        round(r2_3a,            4), round(r2_3b,            4)),
  Adj_R2    = c(NA,                        round(adj_r2_3a,        4), round(adj_r2_3b,        4)),
  AIC       = c(NA,                        round(AIC(fit_trend),   4), round(AIC(fit_ts),      4))
)

cat("\nNote: STL + Random Walk is a benchmark — R², DW, LB, SW are not applicable.\n")
cat("DW ≈ 2 = no autocorrelation | LB p > 0.05 = no autocorrelation | SW p > 0.05 = normal residuals\n\n")
print(diag_comparison)
# ── 7. SAVE OUTPUTS ───────────────────────────────────────────────────────────
cat("\nTraining R-squared (3b)  :", round(summary(fit_ts)$r.squared,    4), "\n")
cat("Adjusted R-squared (3b)  :", round(summary(fit_ts)$adj.r.squared, 4), "\n")
cat("AIC (3b)                 :", round(AIC(fit_ts),                   4), "\n\n")

write.csv(comparison, "lm_metrics.csv", row.names = FALSE)

lm_forecast_df <- data.frame(
  Month    = seq(as.Date(data$Month[n_train + 1]), by = "month", length.out = n_test),
  Forecast = as.numeric(fc_ts_mean),
  Lower95  = as.numeric(fc_ts_lower),
  Upper95  = as.numeric(fc_ts_upper),
  Actual   = as.numeric(test_ts)
)
write.csv(lm_forecast_df, "lm_forecast.csv", row.names = FALSE)

cat("\n=== Linear Regression Analysis Complete ===\n")