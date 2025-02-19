#importy bibliotek
import streamlit as st
import pandas as pd
import json
from pycaret.clustering import load_model, predict_model  
import plotly.express as px  
import os

st.title('Synergy Squad :handshake:')
st.write("---")
#zbiór danych pod zmienną DATA
DATA = 'welcome_survey_simple_v2.csv'
#dane po klasyfikacji
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
#wygenerowane przez AI nazwy i opis
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v0.json'

#mapowanie obrazów
CLUSTER_IMAGES = {
    "Wodni Entuzjaści z Wyższym Wykształceniem": "cluster_0.png",
    "Górscy Profesjonaliści": "cluster_1.png",
    "Koci Miłośnicy Gór": "cluster_2.png",
    "Leśni Eksploratorzy": "cluster_3.png",
    "Górscy Odkrywcy": "cluster_4.png",
    "Wodni Samotnicy":"cluster_5.png",
    "Wodni Profesjonaliści":"cluster_6.png",
    "Wodni Miłośnicy Psów":"cluster_7.png",
    "Leśni Młodzi Profesjonaliści":"cluster_8.png"
}  

#wczytanie modelu
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)
#wczytanie nazw cluster i opisu
@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())
#zaczytanie zbioru danych
@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters


#dodanie pól bocznych sidebarów
with st.sidebar:
    st.header('Znajdź podobnych sobie!')
    st.markdown('Pomożemy Ci znaleść osoby, które mają podobne zainteresowania')
    age = st.selectbox('Wiek', ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([ 
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy: {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
#st.metric("Liczba twoich znajomych", len(same_cluster_df))

total_people = len(all_df)  # pełna liczba osób w grupie
cluster_people = len(same_cluster_df)  # liczba osób w danym klastrze
percentage = (cluster_people / total_people) * 100  # Oblicz procent

# Wyświetlenie kolumn
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Liczba osób w Twojej grupie", cluster_people)

with col2:
    st.metric("Łączna liczba osób ankietowanych", total_people)

with col3:
    st.metric("Procentowy udział Twojej grupy", f"{percentage}%")


#wizualizacje
st.header("Krótka charakterystyka osób z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age",color="age" )
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
    legend_title_text="Wiek",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level", color="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
    legend_title_text="Poziom wykształcenia",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals", color="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
    legend_title_text="Ulubione zwierzęta",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place", color="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
    legend_title_text="Ulubione miejsca",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender", color="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
    legend_title_text="Płeć",
)
    
st.plotly_chart(fig)

st.write("---")
st.header("A teraz sprawdź jak Twoją grupę widzi AI")
# Pole wyboru (selectbox) dla listy grup
selected_cluster = st.selectbox("Wybierz grupę:",  list(CLUSTER_IMAGES.keys()))
# Pobranie ścieżki do obrazu na podstawie wyboru
cluster_image_path = CLUSTER_IMAGES.get(selected_cluster, None)

# Jeśli istnieje obrazek, wyświetl go
if cluster_image_path and os.path.exists(cluster_image_path):
    st.image(cluster_image_path, caption=f"{selected_cluster}", use_container_width =True)
else:
    st.warning(f"Brak obrazu dla wybranej grupy {selected_cluster}. Sprawdź poprawność plików.")