plt.figure(figsize=(12, 4))

# Random Forest
plt.subplot(1, 3, 1)
plt.scatter(y_test, models['Random Forest'].predict(X_test), alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Random Forest')

# KNN
plt.subplot(1, 3, 2)
plt.scatter(y_test, models['KNN'].predict(X_test), alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('KNN')

# Linear Regression
plt.subplot(1, 3, 3)
plt.scatter(y_test, models['Linear Regression'].predict(X_test), alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Linear Regression')

plt.tight_layout()
plt.savefig('all_models_predicted_vs_actual.png')
plt.close()