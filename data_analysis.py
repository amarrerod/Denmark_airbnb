import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly.express as px
import sys

PREPROCESS = "preprocess"

# Display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sns.set(color_codes=True)
sns.set(style="white", palette="muted")


def load_data():
    filepath = "./data/processed_dataset.csv"
    df = pd.read_csv(filepath, parse_dates=True)
    return df


"""
    Funcion empleada para procesar los datos.
    - Nos quedamos solo con las columnas que queremos
    - Transformamos el precio de $cant a float(cant)
    - Guardamos el dataset procesado.
"""


def preprocess_dataset():
    file_path = "./data/listings.csv"
    # Propiedades que vamos a evaluar de los pisos
    features = ['id', 'listing_url', 'name', "host_id", "host_name", "host_location", "street", "neighbourhood", "city",
                "state", "zipcode", "country", "latitude", "longitude", "is_location_exact", "room_type",
                "accommodates", "beds", "bed_type", "price", "weekly_price", "guests_included", "minimum_nights",
                "maximum_nights", "number_of_reviews", "number_of_reviews_ltm", "review_scores_rating",
                "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
                "review_scores_communication", "review_scores_location", "review_scores_value", "has_availability",
                "availability_365"]

    dataframe = pd.read_csv(file_path, usecols=features)
    # Primero, antes de mostrar nada, cambiamos dolares a floats
    dataframe[["price", "weekly_price"]] = dataframe[[
        "price", "weekly_price"]].replace('[\$,]', '', regex=True).astype(float)
    dataframe.weekly_price = dataframe.weekly_price.fillna(-1)
    dataframe = dataframe.dropna()
    # Limpiamos y formateamos las cadenas de ciudad y estado
    dataframe = dataframe[dataframe["city"].map(len) > 4]
    dataframe = dataframe[dataframe["state"].map(len) > 4]
    dataframe = dataframe[dataframe["state"] != "00000"]
    dataframe[["city", "state"]] = dataframe[["city", "state"]].replace('\.', '', regex=True)

    dataframe.state = dataframe["state"].replace("Denmark \nDenmark", "Denmark", regex=False)
    dataframe[["city", "state"]] = dataframe[["city", "state"]].replace('Copenhagenv', 'Copenhagen', regex=False)

    dataframe[["city", "state"]] = dataframe[["city", "state"]].replace('^Kbh [\w]', 'København', regex=True)

    dataframe[["city", "state"]] = dataframe[["city", "state"]].replace('Copenhaguen', 'Copenhagen', regex=False)
    dataframe[["city", "state"]] = dataframe[["city", "state"]].replace('Kopenhagen', 'Copenhagen', regex=False)

    dataframe["city"] = dataframe["city"].str.strip().str.capitalize()
    dataframe["state"] = dataframe["state"].str.strip().str.capitalize()

    dataframe.to_csv("./data/processed_dataset.csv")


def top_best_hosts_in_dk(airbnb_data):
    # Cuales son los hosts con mas pisos
    best_20_hosts = airbnb_data.host_id.value_counts().reset_index().head(20)
    best_20_hosts.columns = ["host_id", "number_of_flats"]
    best_20_hosts = best_20_hosts.merge(airbnb_data[["host_id", "host_name"]], left_on="host_id", right_on="host_id",
                                        how="left", copy=False)
    best_20_hosts.drop_duplicates(inplace=True)
    plt.title("Best 20 Hosts in Denmark")
    plt.bar(best_20_hosts["host_name"], best_20_hosts.number_of_flats)
    plt.xticks(rotation=90)
    plt.ylabel("Number of flats")
    plt.xlabel("Host")
    plt.show()

    # Ahora, nos centraremos en el TOP 10 para ver donde estan sus pisos
    best_10 = airbnb_data.host_id.value_counts().reset_index().head(10)
    best_10.columns = ["host_id", "number_of_flats"]
    best_10 = best_10.merge(airbnb_data[["host_id", "host_name", "price", "latitude", "longitude"]], left_on="host_id",
                            right_on="host_id", how="left", copy=False)
    best_10.drop_duplicates(inplace=True)
    print(f"Top-10 hosts\n{best_10}")
    print(best_10.info())
    # Empleamos Plotly para generar el mapa con los puntos de los pisos
    fig = px.scatter_mapbox(best_10, lat="latitude", lon="longitude", color="host_name", size="price", size_max=30,
                            opacity=.70, zoom=10)
    fig.update_layout(title_text="Top 10 hosts and their flats", height=800)
    fig.layout.mapbox.style = "carto-positron"
    fig.show()


def extract_by_budget(airbnb_data, ascending=False, head=10):
    # Quitamos las opciones de pisos compartidos
    top_flats_by_budget = airbnb_data.sort_values(
        by="price", ascending=ascending).head(head)
    top_flats_by_budget = top_flats_by_budget[["listing_url", "name", "host_id", "host_name", "neighbourhood",
                                               "country", "latitude", "longitude", "room_type", "price"]]

    print(f"Top-{head} flats: \n{top_flats_by_budget}")

    fig = px.scatter_mapbox(top_flats_by_budget, lat="latitude", lon="longitude", color="host_name", size="price",
                            hover_name="listing_url",
                            size_max=30,
                            opacity=.70, zoom=10)
    fig.update_layout(title_text=f"Top {head} flats", height=800)
    fig.layout.mapbox.style = "carto-positron"
    fig.show()


def price_distribution(airbnb_data):
    # Distribucion de los precios
    sns.distplot(airbnb_data["price"], label="Price", norm_hist=False, kde=True)
    sns.distplot(airbnb_data["weekly_price"], label="Weekly price", norm_hist=False, kde=True)
    plt.title("Price distribution")
    plt.xlabel("Price")
    plt.ylabel("Flats")
    plt.legend()
    plt.show()


# Mostramos las ciudades con mas pisos y la media de los precios
def cities_with_more_flats(airbnb_data):
    print(f"4. - Cities: \n{airbnb_data.city.unique()}")
    mean_price = airbnb_data["price"].mean()
    std_price = airbnb_data["price"].std()
    print(f"The mean price is: {mean_price}")
    print(f"Std of the price is: {std_price}")

    # Buscamos los pisos de cada ciudad que estan por encima de la media
    over_mean_flats = airbnb_data[["id", "city", "price"]]
    over_mean_flats.loc[over_mean_flats.price >= mean_price, "over_mean"] = 1
    over_mean_flats.loc[over_mean_flats.price < mean_price, "over_mean"] = 0
    over_mean_flats = over_mean_flats.groupby(by="city").aggregate({"over_mean": "sum"})
    print(f"Over mean flats per city:\n{over_mean_flats}")

    # Numero de pisos por ciudad
    flats_per_cities = airbnb_data.groupby(by="city").aggregate({"id": "count", "price": "mean"})
    flats_per_cities.columns = ["number_of_flats", "mean_price"]
    # Mezclamos con el anterior dataset
    flats_per_cities = flats_per_cities.merge(over_mean_flats[["over_mean"]], left_on="city", right_on="city",
                                              how="left", copy=False)

    flats_per_cities["over_mean_perc"] = flats_per_cities.apply(lambda x: x["over_mean"] / x["number_of_flats"], axis=1)
    flats_per_cities = flats_per_cities.sort_values(by='number_of_flats', ascending=False)
    print(f"Flats per city sorted:\n{flats_per_cities}")

    # Como hay mucha diferencia, para mejorar las graficas nos quedamos con el top5
    flats_per_cities = flats_per_cities.head(5)
    sns.barplot(x=flats_per_cities.index, y=flats_per_cities.number_of_flats)
    plt.title("Top-5 cities with more flats")
    plt.xlabel("City")
    plt.ylabel("Number of Flats")
    plt.xticks(rotation=90)
    plt.show()

    # Grafico con el porcentaje de pisos que superan la media
    sns.barplot(x=flats_per_cities.index, y=flats_per_cities.over_mean_perc)
    plt.title("Percentage of flats over mean in each Top-5 city")
    plt.xlabel("City")
    plt.ylabel("Number of flats")
    plt.xticks(rotation=90)
    plt.show()

    # Grafico de queso
    plt.title("Top-5 cities with more flats")
    plt.pie(flats_per_cities.number_of_flats, labels=flats_per_cities.index, autopct='%1.1f%%', startangle=90,
            pctdistance=0.85)
    # draw circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.show()


def show_correlations(airbnb_data):
    corr_features = ["price", "weekly_price", "guests_included", "minimum_nights", "maximum_nights",
                     "number_of_reviews", "number_of_reviews_ltm", "review_scores_rating",
                     "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
                     "review_scores_communication", "review_scores_location", "review_scores_value", "has_availability",
                     "availability_365", "beds"]
    corr_dataset = airbnb_data[corr_features]
    print(f"Correlation dataset head:\n{corr_dataset.head()}")
    print(corr_dataset.corr())
    sns.pairplot(corr_dataset, palette="Set1")
    plt.show()


def analyse_data(airbnb_data):
    print(f"1. - Info: \n{airbnb_data.info()}")
    print(f"2. - Head: \n{airbnb_data.head()}")
    show_correlations(airbnb_data)
    cities_with_more_flats(airbnb_data)
    price_distribution(airbnb_data)
    top_best_hosts_in_dk(airbnb_data)
    extract_by_budget(airbnb_data, ascending=True, head=30)

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == PREPROCESS:
        preprocess_dataset()
        print("Dataset preprocessed")

    else:
        airbnb_data = load_data()
        analyse_data(airbnb_data)
