# Appartment Prices Prediction

This is an educational project within the MIPT MLOps course.

There is a data set about apartments in the city of Seattle, Washington. The task is to predict the price of housing according to the available data.
The data has the following columns:
* id — housing identification number
* date — date of sale of the house
* price — price
* bedrooms — number of bedrooms
* bathrooms — the number of bathrooms where .5 means a room with a toilet, but without a shower
* sqft_living — housing area
* sqft_lot — plot area
* floors — number of floors
* waterfront — is the embankment visible
* view — how good is the view
* condition — index from 1 to 5, responsible for the condition of the apartment
* grade — 1 to 13, 1-3 corresponds to a poor level of construction and design, 3-7 — medium level, 11-13 — high.
* sqft_above — living area above ground level
* sqft_basement — living area below ground level
* yr_built — year of housing construction
* yr_renovated — the year of the last housing renovation
* zipcode — zip code
* lat — latitude
* long — longitude

# CLI

```
poetry intsall
# train model and save it
# params if needed: --train_data_path=<> --path_to_save=<>
poetry run python3 ./apartment_prices_prediction/train.py

# evaluate model on vaidation data
# params if needed: -val_data_path=<> --model_path=<> --path_to_save_pred=<>
poetry run python3 ./apartment_prices_prediction/infer.py 
```
