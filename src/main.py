import pandas
import statsmodels.formula.api as smf
import graphing

data = {
    'boot_size': [
        39, 38 , 37, 39, 38, 35, 37, 36, 35, 40,
        40, 36, 38, 34, 42, 42, 36, 36, 35, 41,
        42, 38, 37, 35, 40, 36, 35, 39, 41, 37,
        35, 41, 39, 41, 42, 42, 36, 37, 37, 39,
        42, 35, 36, 41, 41, 41, 39, 39, 35, 39,
    ],
    'harness_size': [
        58, 58, 52, 58, 57, 52, 55, 53, 49, 54,
        59, 56, 53, 58, 57, 58, 56, 51, 50, 59,
        59, 59, 55, 50, 55, 52, 53, 54, 51, 56,
        55, 60, 57, 56, 61, 58, 53, 57, 57, 55,
        60, 51, 52, 56, 55, 57, 58, 57, 51, 59,
    ]
}

dataset = pandas.DataFrame(data)

print(dataset)

formula = "boot_size ~ harness_size"

model = smf.ols(formula = formula, data = dataset)

fitted_model = model.fit()

print("The following model parameters have been found:\n" +
      f"Line slope: {fitted_model.params[1]}\n"+
      f"Line Intercept: {fitted_model.params[0]}")

# Show a graph of the result
# Don't worry about how this works for now
graphing.scatter_2D(dataset,    label_x="harness_size", 
                    label_y="boot_size",
                    trendline=lambda x: fitted_model.params[1] * x + fitted_model.params[0]
                    )

# harness_size states the size of the harness we are interested in
harness_size = { 'harness_size' : [52.5] }

# Use the model to predict what size of boots the dog will fit
approximate_boot_size = fitted_model.predict(harness_size)

# Print the result
print("Estimated approximate_boot_size:")
print(approximate_boot_size[0])

