## Development and Implementation Documentation
- Pyhthon 3.12 is recommended
- lucas_organic_carbon_target.csv and lucas_organic_carbon_training_and_test_data.csv have to be located in the same directory as the streamlit_app.py file
- the autoencoder_model.h5 file has to be in a subdirectory "models/" relatively from the streamlit_app.py file
- to run this project use the Python module streamlit like this: "streamlit run streamlit_app.py"

## Purpose and scope
### Milestone1
- Task1: Identify Overall Misclassification Patterns
- Solution: Confusion Matrix

- Task2: Compare True Labels with Predicted Labels to Identify Major Misclassifications
- Solution: Confusion Matrix, PCA Scatterplot

- Task3: Detect Class Imbalance
- Solution: Confusion Matrix, Classification Statistics Table that shows the count of True Positives, False positives and False Negatives

- Task4: Explore classification error for specification groups of data
- Solution: when clicking the confusion matrix a scatter plot with a subset according to selection is rendered

### Milestone2
- Task 1: Show the relative importance of input variables
- Solution: Column Chart with Top Feature Importance Features

- Task 2: Explore the impact of specific values of input variables to the classification process
- Solution: Inspection of Feature Importance of Intervals

- Task 3: Explore interaction between two input variables and their relationships with the classification target 
- Solution: Side-by-side comparison with the same model but different intervals possible

- Task 4: Rank input variables according to their feature importance 
- Solution: Column Chart with Top Feature Importance Features

- Task 5: Selection of input variables and intervals
- Solution: Selection of number of intervals with slider and selection of interval with drop-down menu

### Milestone3
- Task1: Extend current concept of error analysis
- Solution: Side-by-side comparison of Classification results using Confusion Matrix and PCA Scatterplot

- Task2: Extend current concept of feature importance
- Solution: Side-by-side comparison of Feature Importance using Column Chart and drop-down selection

- Task 3: Extend interaction methods
- Solution: Select models to be compared with drop-down menus; ability to open sub menu for specifying parameters of model to be trained; use sliders to determine number of top Features to display or number of intervals to split the   features into

## Components
- Normalization option: done
- Confustion Matrix: done
- scatter plot: done
- classification Statistics: done
- feature importance ranking: done
- feature importance inspection of intervals: done
- options sidebar: done
- in-app model training: done

## Major Implementation Activities
- Normalization option: calculating normalized values and optimize internal logic for turning the option on and off
- Confusion Matrix: picking fitting color scheme, implementing hover information
- scatter plot: implementing PCA, link content to Confusion Matrix selection
- classification Statistics: defining True Positives, False positives and False Negatives
- feature importance inspection of intervals: handling intervals and user interaction when changing point values
