import pandas as pd
from transformers import pipeline
import re


def cargar_datos():
    # Cargar los datasets
    df1 = pd.read_csv('2000-2009 Movies Box Ofice Collection.csv')
    df2 = pd.read_csv('2010-2024 Movies Box Ofice Collection.csv')
    df3 = pd.read_csv('2024 Movies Box Ofice Collection.csv')
    
    # Combinar los dataframes
    df_total = pd.concat([df1, df2, df3], ignore_index=True)
    return df_total

df = cargar_datos()



# Cargar una pipeline de transformers para el procesamiento de preguntas
nlp = pipeline("question-answering")

def buscar_respuesta(df, pregunta):
    # Intentar extraer el nombre de la película de la pregunta
    match = re.search(r"of\s+(.+)\?", pregunta)
    if match:
        nombre_pelicula = match.group(1).strip()
        # Filtrar el DataFrame para obtener solo las filas que contienen el nombre de la película
        df_filtrado = df[df['Release Group'].str.contains(nombre_pelicula, case=False, na=False)]
        if df_filtrado.empty:
            return "No encontré datos para esa película."
        
        # Crear el contexto solo con las entradas relevantes
        context = ' '.join(df_filtrado['Release Group'] + " collected " + df_filtrado['Worldwide'])
        # Hacer la pregunta al modelo
        result = nlp(question="What was the worldwide collection of " + nombre_pelicula + "?", context=context)
        return result['answer']
    else:
        return "Por favor, formula la pregunta de manera específica incluyendo el nombre de la película."



def chatbot_interaction(df):
    while True:
        pregunta = input("Pregúntame sobre la recaudación de cualquier película entre 2000 y 2024 o escribe 'salir' para terminar: ")
        if pregunta.lower() == 'salir':
            print("Gracias por utilizar el chatbot de taquilla de películas. ¡Hasta pronto!")
            break
        respuesta = buscar_respuesta(df, pregunta)
        print(f"Respuesta: {respuesta}")

# Ejecutar la interacción
chatbot_interaction(df)
