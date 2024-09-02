
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wordcloud
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Leer el archivo CSV (asegúrate de que la ruta sea correcta)
df = pd.read_csv("USvideos.csv")

# Llenar valores faltantes en la columna 'description'
df["description"] = df["description"].fillna(value="")


# Función para verificar si una palabra está en mayúscula en el título
def contiene_palabra_en_mayuscula(s):
    for w in s.split():
        if w.isupper():
            return True
    return False


# Aplicar la función y crear nueva columna 'contiene_mayusculas'
df["contiene_mayusculas"] = df["title"].apply(contiene_palabra_en_mayuscula)

# Crear una columna para la longitud del título
df["longitud_titulo"] = df["title"].apply(lambda x: len(x))


# Función para convertir un gráfico de Matplotlib en una imagen de bytes
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return ImageTk.PhotoImage(img)


def mostrar_grafico_pastel():
    # Contar los valores True y False en la columna 'contiene_mayusculas'
    contiene_mayusculas_count = df["contiene_mayusculas"].value_counts()

    # Obtener los conteos específicos
    true_count = contiene_mayusculas_count.get(True, 0)
    false_count = contiene_mayusculas_count.get(False, 0)

    # Configuración de paleta de colores personalizada (azul y morado)
    colors = ["#B0E0E6", "#E0B0FF"]

    # Crear el gráfico de pastel con los valores calculados
    fig, ax = plt.subplots()
    ax.pie(
        [false_count, true_count],
        labels=["No", "Sí"],
        colors=colors,
        textprops={"color": "#000000"},
        startangle=45,
    )
    ax.axis("equal")  # Asegurar que el gráfico de pastel sea circular
    ax.set_title("¿El título contiene palabras en mayúscula?", color="#282828")

    return fig_to_image(fig)


# Función para mostrar el histograma de longitud de títulos
def mostrar_histograma():
    # Crear figura y ejes usando seaborn
    fig, ax = plt.subplots()
    sns.histplot(df["longitud_titulo"], kde=False, color="#ab93bf", bins=30, ax=ax)
    ax.set(
        xlabel="Longitud del Título", ylabel="No. de videos", xticks=range(0, 110, 10)
    )
    ax.set_title("Distribución de la Longitud de los Títulos", color="#282828")
    ax.xaxis.label.set_color("#282828")
    ax.yaxis.label.set_color("#282828")
    ax.tick_params(colors="#282828")

    return fig_to_image(fig)


# Función para mostrar el diagrama de dispersión de vistas vs longitud de títulos
def mostrar_diagrama_dispersion():
    # Crear figura y ejes usando matplotlib
    fig, ax = plt.subplots()
    ax.scatter(
        x=df["views"],
        y=df["longitud_titulo"],
        color="#4d91a0",
        edgecolors="#000000",
        linewidths=0.5,
    )
    ax.set(xlabel="Vistas", ylabel="Longitud del Título")
    ax.set_title(
        "Diagrama de Dispersión: Vistas vs Longitud del Título", color="#282828"
    )
    ax.xaxis.label.set_color("#282828")
    ax.yaxis.label.set_color("#282828")
    ax.tick_params(colors="#282828")

    return fig_to_image(fig)


# Función para mostrar el mapa de calor de correlación
def mostrar_mapa_calor():
    # Obtener etiquetas para los ejes del mapa de calor
    numeric_df = df.select_dtypes(include=[float, int, bool])
    h_labels = [x.replace("_", " ").title() for x in numeric_df.columns]

    # Crear una paleta de colores personalizada en tonos pastel (azules, morados, rosa)
    custom_palette = sns.color_palette(
        ["#a897b8", "#e8bee8", "#c299c0", "#e5bbcb", "#a1d6df"]
    )

    # Crear figura y ejes usando seaborn
    fig, ax = plt.subplots(figsize=(12, 10))  # Ajusta el tamaño de la figura aquí
    heatmap = sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        xticklabels=h_labels,
        yticklabels=h_labels,
        cmap=custom_palette,
        ax=ax,
    )
    ax.set_title("Mapa de Calor de Correlación", color="#282828")

    # Ajustar el diseño automáticamente para evitar que se corten los elementos
    fig.tight_layout()

    return fig_to_image(fig)


# Función para mostrar la nube de palabras de los títulos
def mostrar_nube_palabras():
    # Obtener palabras de los títulos
    title_words = list(df["title"].apply(lambda x: x.split()))
    title_words = [x for y in title_words for x in y]

    # Crear nube de palabras usando WordCloud
    wc = wordcloud.WordCloud(
        width=1200,
        height=500,
        collocations=False,
        background_color="white",
        colormap="Pastel2",
    ).generate(" ".join(title_words))

    # Mostrar la nube de palabras en una ventana nueva
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nube de Palabras de los Títulos de los Videos", color="#282828")

    return fig_to_image(fig)


# Función para encontrar el número óptimo de clusters usando el método del codo
def encontrar_k_optimo(data, max_k=10):
    wcss = []
    fig, ax = plt.subplots(figsize=(10, 8))  # Define fig and ax here

    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # Plot the elbow method graph
    ax.plot(range(1, max_k+1), wcss, marker='o', linestyle='--', color='b')
    ax.set_xlabel('Número de Clusters (k)')
    ax.set_ylabel('WCSS')
    ax.set_title('Método del Codo para Encontrar el k Óptimo')

    return fig_to_image(fig), wcss


# Función para ejecutar KMeans y mostrar los clusters en un scatter plot
def ejecutar_kmeans():
    # Seleccionar las características numéricas para KMeans
    features = ['views', 'likes', 'dislikes', 'comment_count']
    data_for_kmeans = df[features]

    # Encontrar el número óptimo de clusters
    elbow_img, wcss = encontrar_k_optimo(data_for_kmeans, max_k=10)
    
    # Determinar el valor óptimo de k visualmente
    k_optimo = 4  # Debes determinar este valor a partir de la gráfica del método del codo

    # Configurar y entrenar el modelo KMeans con el k óptimo
    kmeans = KMeans(n_clusters=k_optimo, random_state=0)
    kmeans.fit(data_for_kmeans)
    
    # Predecir los clusters para cada punto de datos
    df['cluster'] = kmeans.labels_

    # Crear figura y ejes 3D usando matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotear los puntos con colores según el cluster
    scatter = ax.scatter(df['views'], df['likes'], df['comment_count'], c=df['cluster'], cmap='viridis', alpha=0.7)

    # Añadir leyenda
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Clusters")
    ax.add_artist(legend1)

    # Configurar etiquetas y título
    ax.set_xlabel('Vistas')
    ax.set_ylabel('Likes')
    ax.set_zlabel('Comentarios')
    ax.set_title('Clustering de Videos de YouTube (KMeans en 3D)', color='#282828')

    return fig_to_image(fig)



def clasificar_videos_knn():
    # Seleccionar características y variable objetivo
    features = ['likes', 'dislikes', 'comment_count', 'longitud_titulo']
    target = 'category_id'  # Asegúrate de tener esta columna en tu DataFrame

    X = df[features]
    y = df[target]

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Escalar las características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Crear y entrenar el modelo KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = knn.predict(X_test)

    # Mostrar matriz de confusión y reporte de clasificación
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Devolver el resultado como imagen
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Matriz de Confusión para KNN")
    ax.set_xlabel("Etiqueta Predicha")
    ax.set_ylabel("Etiqueta Verdadera")
    
    return fig_to_image(fig)



def predecir_vistas_arbol_decision():
    # Seleccionar características y variable objetivo
    try:
        # Seleccionar características y variable objetivo
        features = ['likes', 'dislikes', 'comment_count', 'longitud_titulo']
        target = 'views'

        X = df[features]
        y = df[target]

        # Dividir los datos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Crear y entrenar el modelo de árbol de decisión con parámetros para limitar el tamaño del árbol
        tree_reg = DecisionTreeRegressor(random_state=42, max_depth=2, min_samples_split=7, min_samples_leaf=4)
        tree_reg.fit(X_train, y_train)

        # Realizar predicciones
        y_pred = tree_reg.predict(X_test)

        # Mostrar importancia de las características
        fig, ax = plt.subplots(figsize=(12, 8))  # Ajusta el tamaño de la figura aquí
        plot_tree(tree_reg, feature_names=features, filled=True, ax=ax)
        ax.set_title("Árbol de Decisión para Predicción de Vistas")

        return fig_to_image(fig)

    except Exception as e:
        print(f"Error al ejecutar predecir_vistas_arbol_decision: {e}")


# Función para manejar el evento de selección del gráfico
def on_seleccion_grafico(event):
    selected_chart = chart_dropdown.get()
    if selected_chart == "Gráfico de Pastel":
        img = mostrar_grafico_pastel()
    elif selected_chart == "Histograma":
        img = mostrar_histograma()
    elif selected_chart == "Diagrama de Dispersión":
        img = mostrar_diagrama_dispersion()
    elif selected_chart == "Mapa de Calor":
        img = mostrar_mapa_calor()
    elif selected_chart == "Nube de Palabras":
        img = mostrar_nube_palabras()
    elif selected_chart == "Clustering KMeans":
        img = ejecutar_kmeans()
    elif selected_chart == "Clasificación KNN":
        img = clasificar_videos_knn()
    elif selected_chart == "Predicción Árbol de Decisión":
        img = predecir_vistas_arbol_decision()

    label_grafico.config(image=img)
    label_grafico.image = img


# Función para manejar el cierre de la ventana
def al_cerrar():
    root.quit()


# Crear la ventana principal de tkinter
root = tk.Tk()
root.title("Análisis de Videos de YouTube")
root.geometry("1200x900")  # Ajusta el tamaño de la ventana principal aquí

# Configurar el evento de cierre de la ventana
root.protocol("WM_DELETE_WINDOW", al_cerrar)

# Crear un dropdown para seleccionar el gráfico
chart_dropdown = ttk.Combobox(
    root,
    values=[
        "Gráfico de Pastel",
        "Histograma",
        "Diagrama de Dispersión",
        "Mapa de Calor",
        "Nube de Palabras",
        "Clustering KMeans",
        "Clasificación KNN",
        "Predicción Árbol de Decisión"
    ],
    width=30,
)
chart_dropdown.current(0)
chart_dropdown.bind("<<ComboboxSelected>>", on_seleccion_grafico)
chart_dropdown.pack(pady=20)

# Crear una etiqueta para mostrar el gráfico
label_grafico = tk.Label(root)
label_grafico.pack(expand=True)

# Mostrar el gráfico inicial
on_seleccion_grafico(None)

# Ejecutar la aplicación
root.mainloop() 