
#Librerias 
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu 
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import base64
import pydeck as pdk
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Ajustamos las medidas de la pantalla para el dashboard
st.markdown("""
    <style>
    .block-container {
        max-width: 1200px;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 2rem;
    }

    /* Estilo para el sidebar */
    section[data-testid="stSidebar"] {
        background-color: #A9DGE5;  
        color: #2C3E50;
        padding: 20px;
        border-right: 2px solid #560B28D
    }

    /* Cambiar color de los títulos en sidebar */
    .sidebar .block-container h1, 
    .sidebar .block-container h2, 
    .sidebar .block-container h3 {
        color: #2C3E50;
    }

    /* Cambiar color de los labels de los widgets */
    .css-1r6slb0 p {
        color: #2C3E50;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)



# Función para cargar imagen local como base64
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_img_as_base64("./img/LOGO.png")

components.html(f"""
    <style>
    .custom-header {{
        padding: 15px 50px;
        position: fixed;
        top: 0;
        left: 0;
        height: 140px;
        width: 100vw;
        z-index: 9999;
        display: flex;
        align-items: center;
        box-shadow: 0 1px 5px rgba(0,0,0,0.1);
    }}

    .custom-header img {{
        height: 140px;
        margin-right: 20px;
    }}

    .custom-header h1 {{
        font-size: 80px;
        font-weight: 700;
        color: #2C3E50;
        margin: 0;
        font-family: fangsong;
    }}

    .stApp {{
        margin-top: 160px;
    }}
    </style>

    <div class="custom-header">
        <img src="data:image/png;base64,{img_base64}">
        <h1>GrowthTrack</h1>
    </div>
""", height=140)

@st.cache_resource
def load_data():
    #carga de archivos
    dQ = pd.read_csv('./dataset/Quebec_limpio.csv')
    dM = pd.read_csv('./dataset/Malta_limpio.csv')
    dV = pd.read_csv('./dataset/Victoria_limpio.csv')
    dMx = pd.read_csv('./dataset/Mexico_limpio.csv')

    #archivos limpios
    dQ=dQ.drop('Unnamed: 0', axis = 1)
    dM=dM.drop('Unnamed: 0', axis = 1)
    dV=dV.drop('Unnamed: 0', axis = 1)
    dMx=dMx.drop('Unnamed: 0', axis = 1)
    
    #Cambiamos object por numeric
    dQ['host_identity_verified'] = dQ['host_identity_verified'].replace({'f': 0, 't': 1})
    dQ['instant_bookable'] = dQ['instant_bookable'].replace({'f': 0, 't': 1})

    dQ['host_is_superhost'] = dQ['host_is_superhost'].replace({'f': 0, 't': 1})

    dQ['host_response_rate'] = dQ['host_response_rate'].astype(str).str.rstrip('%')
    dQ['host_response_rate'] = pd.to_numeric(dQ['host_response_rate'], errors='coerce').fillna(0)

    dQ['room_type'] = dQ['room_type'].map({
        'Entire home/apt': 1,
        'Private room': 2,
        'Shared room': 3,
        'Hotel room': 4
    })

    
    dM['host_identity_verified'] = dM['host_identity_verified'].replace({'f': 0, 't': 1})
    dM['instant_bookable'] = dM['instant_bookable'].replace({'f': 0, 't': 1})
    dM['host_is_superhost'] = dM['host_is_superhost'].replace({'f': 0, 't': 1})

    dM['host_response_rate'] = dM['host_response_rate'].astype(str).str.rstrip('%')
    dM['host_response_rate'] = pd.to_numeric(dM['host_response_rate'], errors='coerce').fillna(0)

    dM['room_type'] = dM['room_type'].map({
        'Entire home/apt': 1,
        'Private room': 2,
        'Shared room': 3,
        'Hotel room': 4
    })

    
    dV['host_identity_verified'] = dV['host_identity_verified'].replace({'f': 0, 't': 1})
    dV['instant_bookable'] = dV['instant_bookable'].replace({'f': 0, 't': 1})
    dV['host_is_superhost'] = dV['host_is_superhost'].replace({'f': 0, 't': 1})

    dV['host_response_rate'] = dV['host_response_rate'].astype(str).str.rstrip('%')
    dV['host_response_rate'] = pd.to_numeric(dV['host_response_rate'], errors='coerce').fillna(0)

    dV['room_type'] = dV['room_type'].map({
        'Entire home/apt': 1,
        'Private room': 2,
        'Shared room': 3,
        'Hotel room': 4
    })

    
    dMx['host_identity_verified'] = dMx['host_identity_verified'].replace({'f': 0, 't': 1})
    dMx['instant_bookable'] = dMx['instant_bookable'].replace({'f': 0, 't': 1})
    dMx['host_is_superhost'] = dMx['host_is_superhost'].replace({'f': 0, 't': 1})

    dMx['host_response_rate'] = dMx['host_response_rate'].astype(str).str.rstrip('%')
    dMx['host_response_rate'] = pd.to_numeric(dMx['host_response_rate'], errors='coerce').fillna(0)

    dMx['room_type'] = dMx['room_type'].map({
        'Entire home/apt': 1,
        'Private room': 2,
        'Shared room': 3,
        'Hotel room': 4
    })

    numeric_df = dMx.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns

    text_df = dMx.select_dtypes(['object'])
    text_cols = text_df.columns

    numeric_df = dQ.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns

    text_df = dQ.select_dtypes(['object'])
    text_cols = text_df.columns


    numeric_df = dV.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns

    text_df = dV.select_dtypes(['object'])
    text_cols = text_df.columns

    numeric_df = dM.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns

    text_df = dM.select_dtypes(['object'])
    text_cols = text_df.columns

    numeric_cols2 = list(numeric_cols) 



    return dQ, dM, dV, dMx, numeric_cols, text_cols, numeric_cols2

dQ, dM, dV, dMx, numeric_cols, text_cols, numeric_cols2 = load_data()

#Sidebar 
st.sidebar.image("./img/LOGO.png")
with st.sidebar: 
    pagina = option_menu(
        menu_title = None,
        options = ["Home Page","Lugares", "Comparacion", "Mapa"],
        icons = ["house -fill","cursor", "collection", "geo-alt"],
        default_index = 0, 
        orientation = "vertical",
        styles = {
            "container" : {"padding": "5px", "background-color": "#B7CDE2"},
            "icon" : {"color": "#2C3E50", "font-size" : "40px"},
            "nav-link": {
                "font-size": "0px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#B39CD0", 
            },
            "nav-link-selected": {"background-color": "#B7CDE2"},
        }
    )

#Home Page
if pagina == "Home Page": 
    st.image("./img/GROWTHTRACK.png", use_container_width=True)
    st.markdown("Aquí puedes explorar diversos análisis de datos de la ciudad seleccionada. Visualiza tendencias, distribuciones y aplica modelos predictivos fácilmente.")

if pagina == "Lugares": 
    col1, col2, col3 = st.columns([1,2,3])

    with col1:
        st.markdown("### Selección de sitio")

    with col2:
        pais=st.selectbox("",["México", "Canada", "Italia"])

    with col3:
        rutas = {
            "México": ["CDMX"],
            "Canada": ["Victoria", "Quebec"],
            "Italia": ["Malta"]
        }
        ciudad=st.selectbox("", rutas[pais])


    if pais == "México" and ciudad == "CDMX":
        st.title("CDMX, México")

        sub_opcion = option_menu(
            menu_title=None,
            options=["DataBase", "Univariado", "Regresiones"],
            icons=["table", "bar-chart-line", "activity"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#DCEAF4"},
                "icon": {"color": "#2C3E50", "font-size": "25px"},
                "nav-link": {
                    "font-size": "16px",
                    "margin": "5px",
                    "--hover-color": "#BEE3F8"
                },
                "nav-link-selected": {"background-color": "#B7CDE2"},
            }
        )

        if sub_opcion == "DataBase": 
            st.title("Información relevante de hospedajes")

            col1, col2, col3 = st.columns(3)

            # Valores por defecto
            precio_min = int(dMx["price"].min())
            precio_max = int(dMx["price"].max())
            room_types = dMx["room_type"].unique().tolist()
            default_columns = ["room_type", "price", "neighbourhood_cleansed"]

            # Estado inicial
            if "precio_filtro" not in st.session_state:
                st.session_state["precio_filtro"] = (precio_min, precio_max)
            if "room_filtro" not in st.session_state:
                st.session_state["room_filtro"] = room_types
            if "cols_filtro" not in st.session_state:
                st.session_state["cols_filtro"] = default_columns

            # Cargamos valores desde el estado
            precio_val = st.session_state["precio_filtro"]
            room_val = st.session_state["room_filtro"]
            cols_val = st.session_state["cols_filtro"]

            with col1:  
                st.markdown("### Filtros de búsqueda")  
            with col2: 
                room_val = st.multiselect("Filtrar por tipo de cuarto", options=room_types, default=room_val)
            with col3: 
                cols_val = st.multiselect("Selecciona columnas a visualizar", options=dMx.columns.tolist(), default=cols_val)

            # Guardamos valores nuevamente al estado (para siguiente renderizado)
            st.session_state["precio_filtro"] = precio_val
            st.session_state["room_filtro"] = room_val
            st.session_state["cols_filtro"] = cols_val

            # Aplicar filtros al DataFrame
            dMx_filtrado = dMx[
                (dMx["room_type"].isin(room_val))
            ]

            # Mostrar DataFrame filtrado y resumen
            st.markdown(f"### Datos filtrados ({dMx_filtrado.shape[0]} filas)")
            st.dataframe(dMx_filtrado[cols_val], use_container_width=True)

            st.markdown("### Resumen estadístico")
            st.dataframe(dMx_filtrado.describe(), use_container_width=True)
        
        if sub_opcion == "Univariado": 
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribución Superhosts")
                freq_data = dMx['host_is_superhost'].value_counts().reset_index()
                freq_data.columns = ['category', 'count']
                
                figure1 = px.bar(
                    freq_data,
                    x='category',
                    y='count',
                    title="Superhosts (0=No, 1=Sí)",
                    labels={'category': '', 'count': 'Frecuencia'},
                    color='category',
                    color_discrete_map={0: '#6d38aa', 1: '#fbb77c'}
                )
                st.plotly_chart(figure1, use_container_width=True)
                
            with col2:
                st.subheader("Distribución Precios")
                # Solución directa sin usar histograma intermedio
                price_counts = dMx['price'].value_counts().sort_index().reset_index()
                price_counts.columns = ['price', 'count']
                
                figure2 = px.line(
                    price_counts,
                    x='price',
                    y='count',
                    title=" ",
                    labels={'price': 'Precio', 'count': 'Frecuencia'}
                )
                st.plotly_chart(figure2, use_container_width=True)

            with col1:
                st.subheader("Distribución de Tipos de Habitación")
                room_counts = dMx['room_type'].value_counts().reset_index()
                room_counts.columns = ['room_type', 'count']
                
                figure3 = px.pie(
                    room_counts,
                    names='room_type',
                    values='count',
                    title="Tipo de Habitación:",
                    subtitle="1.-Entire home/apt, 2.-Private room, 3.-Shared room, 4.-Hotel room.",
                )
                st.plotly_chart(figure3, use_container_width=True)

            with  col2:
                # 2. Distribución de Reviews (Scatterplot)
                st.subheader("Puntuaciones de Reviews")
                review_freq = dMx['review_scores_rating'].value_counts().reset_index()
                fig2 = px.scatter(
                    review_freq,
                    x='review_scores_rating',
                    y='count',
                    size='count',
                    color='count',
                    title="Frecuencia de Puntuaciones",
                    labels={'review_scores_rating': 'Puntuación', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                    
            #Tabla de frecuencias
            st.subheader("Tablas Resumen de Frecuencias por Variable")
            # Variables categóricas de interés
            variables_frecuencia = ['host_is_superhost', 'room_type', 'bedrooms', 'review_scores_rating']

            for var in variables_frecuencia:
                freq = dMx[var].value_counts().reset_index()
                freq.columns = ['Valor', 'Frecuencia']
                freq = freq[freq['Frecuencia'] > 50]
                freq['Porcentaje'] = (freq['Frecuencia'] / freq['Frecuencia'].sum()) * 100
                freq['Variable'] = var

                st.markdown(f"#### Variable: {var}")
                st.dataframe(
                    freq[[ 'Valor', 'Frecuencia', 'Porcentaje']],
                    use_container_width=True,
                    hide_index=True
                )


        if sub_opcion == "Regresiones":
            st.title("Modelados Predictivos")
                
            def crear_donut_metrica(valor, etiqueta, color="#83C5BE"):
                # Si el valor es mayor a 1, lo escalamos para que se vea bien en el donut
                if valor > 1:
                    # Usamos un valor máximo de referencia como el múltiplo de 10 más cercano
                    max_valor = round(valor, -1) if valor < 100 else round(valor, -2)
                    proporcion = min(valor / max_valor, 1)
                else:
                    proporcion = valor  # ya está en escala 0 a 1

                fig = go.Figure(go.Pie(
                    values=[proporcion, 1 - proporcion],
                    labels=["", ""],
                    hole=0.7,
                    marker_colors=[color, "#F0F0F0"],
                    textinfo='none'
                ))
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    annotations=[dict(
                        text=f"<b>{etiqueta}</b><br><span style='font-size:24px'>{valor:.2f}</span>",
                        x=0.5, y=0.5, font_size=20, showarrow=False, align='center'
                    )],
                    width=220, height=220
                )
                return fig

            #heatmap de correlación
            # Limpiar duplicados de columnas, si los hubiera
            dMx = dMx.loc[:, ~dMx.columns.duplicated()]
            numeric_cols = dMx.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if 'price' in dMx.columns:
                st.subheader("Heatmap de variables con alta correlación con 'price'")
                # Calcular correlación con 'price'
                corr_matrix = dMx[numeric_cols].corr()
                correlaciones_con_price = corr_matrix['price'].drop('price').abs()
                
                columnas_filtradas = correlaciones_con_price[correlaciones_con_price >= 0.3].index.tolist()
                columnas_heatmap = columnas_filtradas + ['price']
                
                # Generar heatmap
                fig = px.imshow(
                    dMx[columnas_heatmap].corr().abs().round(2),
                    text_auto=True,
                    color_continuous_scale=["#F5F5F5", "#B39CD0", "#6A3FA0"]
                )
                fig.update_layout(width=900, height=500)
                st.plotly_chart(fig, use_container_width=True)

            vista=st.selectbox("Selecciona el tipo de regresión que deseas visualizar", ["Lineal Simple", "Lineal Múltiple", "Logística"])

            if vista == "Lineal Simple":
                st.subheader("Regresión Lineal Simple")

                # Tabs por tipo de cuarto codificado
                tabs = st.tabs(["Entire home/apt", "Private room", "Shared room", "Hotel room"])
                tipos = {
                    "Entire home/apt": 1,
                    "Private room": 2,
                    "Shared room": 3,
                    "Hotel room": 4
                }

                # Generar lista de columnas numéricas limpias
                numeric_cols = dMx.select_dtypes(include=['int64', 'float64']).columns.tolist()
                columnas_a_excluir = [
                    "price",
                    "calculated_host_listings_count",
                    "calculated_host_listings_count_entire_homes",
                    "calculated_host_listings_count_private_rooms",
                    "calculated_host_listings_count_shared_rooms"
                ]
                numeric_cols_limpias = [col for col in numeric_cols if col not in columnas_a_excluir]

                for nombre_tab, tipo_valor in zip(tipos.keys(), tabs):
                    with tipo_valor:
                        df_tipo = dMx[dMx["room_type"] == tipos[nombre_tab]].dropna(subset=["price"])

                        if df_tipo.shape[0] < 2:
                            st.warning("No hay suficientes datos para este tipo de cuarto.")
                            continue

                        # Correlaciones con price
                        # Evaluar R² de regresión lineal simple para cada variable
                        top_vars = {}
                        for col in numeric_cols_limpias:
                            try:
                                df_tmp = df_tipo.dropna(subset=["price", col])
                                if df_tmp.shape[0] < 2:
                                    continue
                                X_tmp = df_tmp[[col]]
                                y_tmp = df_tmp["price"]
                                model_tmp = LinearRegression().fit(X_tmp, y_tmp)
                                r2 = model_tmp.score(X_tmp, y_tmp)
                                if r2 > 0.01:  # umbral bajo para evitar ruido
                                    top_vars[col] = r2
                            except:
                                pass

                        # Ordenar por R² descendente y quedarte con top 5
                        top_vars = dict(sorted(top_vars.items(), key=lambda x: x[1], reverse=True)[:5])


                        if not top_vars:
                            st.warning("No se encontraron variables con buen ajuste para regresión lineal.")
                            continue


                        st.markdown("### Selecciona variable independiente")
                        seleccion = st.selectbox(
                            f"Variable para predecir el precio en {nombre_tab}:",
                            options=list(top_vars.keys()),
                            key=f"{nombre_tab}_selector"
                        )

                        df_valid = df_tipo.dropna(subset=[seleccion])
                        X = df_valid[[seleccion]]
                        y = df_valid["price"]

                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)

                        coef_deter = model.score(X, y)
                        coef_correl = np.sqrt(coef_deter) if coef_deter >= 0 else 0

                    
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown("### Dispersión de datos reales")
                            fig_disp = px.scatter()
                            fig_disp.add_scatter(x=df_valid["room_type"], y=df_valid["price"], mode="markers",
                                                name="room_type vs price", marker=dict(color="#78B6C3"))
                            fig_disp.add_scatter(x=df_valid[seleccion], y=df_valid["price"], mode="markers",
                                                name=f"{seleccion} vs price", marker=dict(color="#3E7D95"))
                            fig_disp.update_layout(
                                title=f"{nombre_tab}: Dispersión de room_type y {seleccion} vs price",
                                xaxis_title="room_type (codificada)",
                                yaxis_title="price"
                            )
                            st.plotly_chart(fig_disp, use_container_width=True)

                            st.markdown("### Gráfico de Regresión")
                            fig = px.scatter(x=df_valid[seleccion], y=y, labels={"x": seleccion, "y": "price"},
                                            title=f"{nombre_tab}: Real vs Predicho")
                            fig.add_scatter(x=df_valid[seleccion], y=y_pred, mode="lines", name="Predicción", line=dict(color="#3E7D95"))
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.markdown("### Métricas del modelo")
                            st.plotly_chart(crear_donut_metrica(model.intercept_, "Intercepto"), use_container_width=True, key=f"{nombre_tab}_intercepto")
                            st.plotly_chart(crear_donut_metrica(coef_deter, "R² (Determinación)"), use_container_width=True, key=f"{nombre_tab}_r2")
                            st.plotly_chart(crear_donut_metrica(coef_correl, "r (Correlación)"), use_container_width=True, key=f"{nombre_tab}_r")

            if vista == "Lineal Múltiple": 
                st.subheader("Regresión Lineal Múltiple")

                # Variables numéricas sin 'price'
                numeric_cols = dMx.select_dtypes(include=['int64', 'float64']).columns.tolist()
                opciones_independientes = [col for col in numeric_cols if col != 'price']

                seleccionadas = st.multiselect("Entrena tu modelo:", options=opciones_independientes)

                if seleccionadas:
                    # Variables
                    Vars_Indep = dMx[seleccionadas].dropna()
                    Var_Dep = dMx.loc[Vars_Indep.index, "price"]

                    # Modelo
                    model = LinearRegression()
                    model.fit(X=Vars_Indep, y=Var_Dep)
                    y_pred = model.predict(X=Vars_Indep)

                    # En esta sección se determinar la mejor variable simple para graficar
                    mejor_var = None
                    mejor_r2 = 0
                    for var in seleccionadas:
                        try:
                            X_temp = Vars_Indep[[var]]
                            model_temp = LinearRegression().fit(X_temp, Var_Dep)
                            r2 = model_temp.score(X_temp, Var_Dep)
                            if r2 > mejor_r2:
                                mejor_r2 = r2
                                mejor_var = var
                        except:
                            continue

                    coef_Deter = model.score(Vars_Indep, Var_Dep)
                    coef_Correl = np.sqrt(coef_Deter) if coef_Deter >= 0 else 0

                    # DataFrame para gráficos
                    f1 = Vars_Indep.copy()
                    f1["price"] = Var_Dep
                    f1["Pred_price"] = y_pred

                    # Layout: gráficos izquierda, métricas derecha
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown("### Gráfico comparativo real vs predicho (una variable)")

                        if mejor_var:
                            x1 = mejor_var
                            x2 = seleccionadas[0] if mejor_var != seleccionadas[0] else (seleccionadas[1] if len(seleccionadas) > 1 else mejor_var)
                            fig1, ax1 = plt.subplots()
                            sns.scatterplot(x=x1, y=x2, color='green', data=f1, label='Real', ax=ax1)
                            sns.scatterplot(x=x1, y=f1["Pred_price"], color='pink', label='Predicho', data=f1, ax=ax1)
                            ax1.set_title("Relación simple entre variables")
                            st.pyplot(fig1)

                        st.markdown("### Gráfico real vs predicho del modelo múltiple")
                        fig2, ax2 = plt.subplots()
                        sns.scatterplot(x="price", y="price", data=f1, label="Real", color="blue", ax=ax2)
                        sns.scatterplot(x="price", y="Pred_price", data=f1, label="Predicho", color="red", ax=ax2)
                        ax2.set_title("Real vs Predicho")
                        st.pyplot(fig2)

                    with col2:
                        st.markdown("### Métricas del modelo")
                        st.plotly_chart(crear_donut_metrica(model.intercept_, "Intercepto"), use_container_width=True, key="intercepto_mult")
                        st.plotly_chart(crear_donut_metrica(coef_Deter, "R² (Determinación)"), use_container_width=True, key="r2_mult")
                        st.plotly_chart(crear_donut_metrica(coef_Correl, "r (Correlación)"), use_container_width=True, key="r_mult")

                else:
                    st.info("Selecciona al menos una variable para entrenar el modelo.")



            if vista == "Logística":
                st.subheader("Regresión Logística")

                # Copia de trabajo del DataFrame
                df_log = dMx.copy()

                # Detectar variables dicotómicas
                dicotomicas = [col for col in df_log.columns if df_log[col].nunique() == 2]
                if "room_type" in df_log.columns:
                    dicotomicas.append("room_type")
                dicotomicas = list(set(dicotomicas))

                # Variables independientes candidatas
                numeric_cols = df_log.select_dtypes(include=['int64', 'float64']).columns.tolist()

                # Selección de variable dependiente
                var_dep = st.selectbox("Selecciona la variable dependiente dicotómica:", options=dicotomicas)

                # Variables independientes 
                opciones_indep = [col for col in numeric_cols if col != var_dep]
                vars_indep = st.multiselect("Selecciona las variables independientes:", options=opciones_indep)

                if var_dep and vars_indep:
                    df_model = df_log.dropna(subset=vars_indep + [var_dep])
                    X = df_model[vars_indep]

                    # Conversión de variable dependiente
                    if var_dep == "room_type":
                        y = df_model["room_type"].apply(lambda x: 1 if x == 1 else 0)  # 1 = Entire
                    else:
                        y = df_model[var_dep]
                        if sorted(y.unique()) != [0, 1]:
                            y = y.map({y.unique()[0]: 0, y.unique()[1]: 1})

                    # División de datos
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # Escalado
                    escalar = StandardScaler()
                    X_train = escalar.fit_transform(X_train)
                    X_test = escalar.transform(X_test)

                    # Modelo
                    modelo = LogisticRegression(max_iter=1000)
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)

                    # Métricas
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    precision = precision_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
                    sensibilidad = recall_score(y_test, y_pred)

                    cm_ordenada = np.array([
                        [cm[1, 1], cm[1, 0]],
                        [cm[0, 1], cm[0, 0]]
                    ])

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("### Matriz de Confusión")
                        fig_cm, ax = plt.subplots()
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm_ordenada)
                        disp.plot(cmap="Purples", ax=ax, colorbar=True)
                        ax.set_title("Matriz de Confusión", fontsize=14)

                        # Etiquetas explicativas
                        ax.text(-0.35, -0.35, "VP\nVerdaderos Positivos", fontsize=8, color="white", weight="bold")
                        ax.text(0.65, -0.35, "FP\nFalsos Positivos", fontsize=8, color="black", weight="bold")
                        ax.text(-0.35, 0.65, "FN\nFalsos Negativos", fontsize=8, color="black", weight="bold")
                        ax.text(0.65, 0.65, "VN\nVerdaderos Negativos", fontsize=8, color="green", weight="bold")

                        st.pyplot(fig_cm)

                    with col2:
                        st.markdown("### Métricas del modelo")
                        st.plotly_chart(crear_donut_metrica(precision, "Precisión"), use_container_width=True, key="log_prec")
                        st.plotly_chart(crear_donut_metrica(accuracy, "Exactitud"), use_container_width=True, key="log_acc")
                        st.plotly_chart(crear_donut_metrica(sensibilidad, "Sensibilidad"), use_container_width=True, key="log_sens")
                else:
                    st.info("Selecciona una variable dependiente dicotómica y al menos una independiente.")

###########################################################################################################################################################
    if pais == "Canada" and ciudad == "Quebec": 
        st.title("Quebec, Canada")

        sub_opcion = option_menu(
            menu_title=None,
            options=["DataBase", "Univariado", "Regresiones"],
            icons=["table", "bar-chart-line", "activity"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#DCEAF4"},
                "icon": {"color": "#2C3E50", "font-size": "25px"},
                "nav-link": {
                    "font-size": "16px",
                    "margin": "5px",
                    "--hover-color": "#BEE3F8"
                },
                "nav-link-selected": {"background-color": "#B7CDE2"},
            }
        )

        if sub_opcion == "DataBase": 
            st.title("Información relevante de hospedajes")

            col1, col2, col3 = st.columns(3)

            # Valores por defecto
            precio_min = int(dQ["price"].min())
            precio_max = int(dQ["price"].max())
            room_types = dQ["room_type"].unique().tolist()
            default_columns = ["room_type", "price", "neighbourhood_cleansed"]

            # Estado inicial
            if "precio_filtro" not in st.session_state:
                st.session_state["precio_filtro"] = (precio_min, precio_max)
            if "room_filtro" not in st.session_state:
                st.session_state["room_filtro"] = room_types
            if "cols_filtro" not in st.session_state:
                st.session_state["cols_filtro"] = default_columns

            # Cargamos valores desde el estado
            precio_val = st.session_state["precio_filtro"]
            room_val = st.session_state["room_filtro"]
            cols_val = st.session_state["cols_filtro"]

            with col1:  
                st.markdown("### Filtros de búsqueda")  
            with col2: 
                room_val = st.multiselect("Filtrar por tipo de cuarto", options=room_types, default=room_val)
            with col3: 
                cols_val = st.multiselect("Selecciona columnas a visualizar", options=dQ.columns.tolist(), default=cols_val)

            # Guardamos valores nuevamente al estado (para siguiente renderizado)
            st.session_state["precio_filtro"] = precio_val
            st.session_state["room_filtro"] = room_val
            st.session_state["cols_filtro"] = cols_val

            # Aplicar filtros al DataFrame
            dQ_filtrado = dQ[
                (dQ["room_type"].isin(room_val))
            ]

            # Mostrar DataFrame filtrado y resumen
            st.markdown(f"### Datos filtrados ({dQ_filtrado.shape[0]} filas)")
            st.dataframe(dQ_filtrado[cols_val], use_container_width=True)

            st.markdown("### Resumen estadístico")
            st.dataframe(dQ_filtrado.describe(), use_container_width=True)
        
        if sub_opcion == "Univariado": 
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribución Superhosts")
                freq_data = dQ['host_is_superhost'].value_counts().reset_index()
                freq_data.columns = ['category', 'count']
                
                figure1 = px.bar(
                    freq_data,
                    x='category',
                    y='count',
                    title="Superhosts (0=No, 1=Sí)",
                    labels={'category': '', 'count': 'Frecuencia'},
                    color='category',
                    color_discrete_map={0: '#6d38aa', 1: '#fbb77c'}
                )
                st.plotly_chart(figure1, use_container_width=True)
                
            with col2:
                st.subheader("Distribución Precios")
                # Solución directa sin usar histograma intermedio
                price_counts = dQ['price'].value_counts().sort_index().reset_index()
                price_counts.columns = ['price', 'count']
                
                figure2 = px.line(
                    price_counts,
                    x='price',
                    y='count',
                    title=" ",
                    labels={'price': 'Precio', 'count': 'Frecuencia'}
                )
                st.plotly_chart(figure2, use_container_width=True)

            with col1:
                st.subheader("Distribución de Tipos de Habitación")
                room_counts = dQ['room_type'].value_counts().reset_index()
                room_counts.columns = ['room_type', 'count']
                
                figure3 = px.pie(
                    room_counts,
                    names='room_type',
                    values='count',
                    title="Tipo de Habitación:",
                    subtitle="1.-Entire home/apt, 2.-Private room, 3.-Shared room, 4.-Hotel room.",
                )
                st.plotly_chart(figure3, use_container_width=True)

            with  col2:
                # 2. Distribución de Reviews (Scatterplot)
                st.subheader("Puntuaciones de Reviews")
                review_freq = dQ['review_scores_rating'].value_counts().reset_index()
                fig2 = px.scatter(
                    review_freq,
                    x='review_scores_rating',
                    y='count',
                    size='count',
                    color='count',
                    title="Frecuencia de Puntuaciones",
                    labels={'review_scores_rating': 'Puntuación', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                    
            #Tabla de frecuencias
            st.subheader("Tablas Resumen de Frecuencias por Variable")
            # Variables categóricas de interés
            variables_frecuencia = ['host_is_superhost', 'room_type', 'bedrooms', 'review_scores_rating']

            for var in variables_frecuencia:
                freq = dQ[var].value_counts().reset_index()
                freq.columns = ['Valor', 'Frecuencia']
                freq = freq[freq['Frecuencia'] > 50]
                freq['Porcentaje'] = (freq['Frecuencia'] / freq['Frecuencia'].sum()) * 100
                freq['Variable'] = var

                st.markdown(f"#### Variable: {var}")
                st.dataframe(
                    freq[[ 'Valor', 'Frecuencia', 'Porcentaje']],
                    use_container_width=True,
                    hide_index=True
                )


        if sub_opcion == "Regresiones":
            st.title("Modelados Predictivos")
                
            def crear_donut_metrica(valor, etiqueta, color="#83C5BE"):
                # Si el valor es mayor a 1, lo escalamos para que se vea bien en el donut
                if valor > 1:
                    # Usamos un valor máximo de referencia como el múltiplo de 10 más cercano
                    max_valor = round(valor, -1) if valor < 100 else round(valor, -2)
                    proporcion = min(valor / max_valor, 1)
                else:
                    proporcion = valor  # ya está en escala 0 a 1

                fig = go.Figure(go.Pie(
                    values=[proporcion, 1 - proporcion],
                    labels=["", ""],
                    hole=0.7,
                    marker_colors=[color, "#F0F0F0"],
                    textinfo='none'
                ))
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    annotations=[dict(
                        text=f"<b>{etiqueta}</b><br><span style='font-size:24px'>{valor:.2f}</span>",
                        x=0.5, y=0.5, font_size=20, showarrow=False, align='center'
                    )],
                    width=220, height=220
                )
                return fig

            #heatmap de correlación
            # Limpiar duplicados de columnas, si los hubiera
            dQ = dQ.loc[:, ~dQ.columns.duplicated()]
            numeric_cols = dQ.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if 'price' in dQ.columns:
                st.subheader("Heatmap de variables con alta correlación con 'price'")
                # Calcular correlación con 'price'
                corr_matrix = dQ[numeric_cols].corr()
                correlaciones_con_price = corr_matrix['price'].drop('price').abs()
                
                columnas_filtradas = correlaciones_con_price[correlaciones_con_price >= 0.3].index.tolist()
                columnas_heatmap = columnas_filtradas + ['price']
                
                # Generar heatmap
                fig = px.imshow(
                    dQ[columnas_heatmap].corr().abs().round(2),
                    text_auto=True,
                    color_continuous_scale=["#F5F5F5", "#B39CD0", "#6A3FA0"]
                )
                fig.update_layout(width=900, height=500)
                st.plotly_chart(fig, use_container_width=True)

            vista=st.selectbox("Selecciona el tipo de regresión que deseas visualizar", ["Lineal Simple", "Lineal Múltiple", "Logística"])

            if vista == "Lineal Simple":
                st.subheader("Regresión Lineal Simple")

                # Tabs por tipo de cuarto codificado
                tabs = st.tabs(["Entire home/apt", "Private room", "Shared room", "Hotel room"])
                tipos = {
                    "Entire home/apt": 1,
                    "Private room": 2,
                    "Shared room": 3,
                    "Hotel room": 4
                }

                # Generar lista de columnas numéricas limpias
                numeric_cols = dQ.select_dtypes(include=['int64', 'float64']).columns.tolist()
                columnas_a_excluir = [
                    "price",
                    "calculated_host_listings_count",
                    "calculated_host_listings_count_entire_homes",
                    "calculated_host_listings_count_private_rooms",
                    "calculated_host_listings_count_shared_rooms"
                ]
                numeric_cols_limpias = [col for col in numeric_cols if col not in columnas_a_excluir]

                for nombre_tab, tipo_valor in zip(tipos.keys(), tabs):
                    with tipo_valor:
                        df_tipo = dQ[dQ["room_type"] == tipos[nombre_tab]].dropna(subset=["price"])

                        if df_tipo.shape[0] < 2:
                            st.warning("No hay suficientes datos para este tipo de cuarto.")
                            continue

                        # Correlaciones con price
                        # Evaluar R² de regresión lineal simple para cada variable
                        top_vars = {}
                        for col in numeric_cols_limpias:
                            try:
                                df_tmp = df_tipo.dropna(subset=["price", col])
                                if df_tmp.shape[0] < 2:
                                    continue
                                X_tmp = df_tmp[[col]]
                                y_tmp = df_tmp["price"]
                                model_tmp = LinearRegression().fit(X_tmp, y_tmp)
                                r2 = model_tmp.score(X_tmp, y_tmp)
                                if r2 > 0.01:  # umbral bajo para evitar ruido
                                    top_vars[col] = r2
                            except:
                                pass

                        # Ordenar por R² descendente y quedarte con top 5
                        top_vars = dict(sorted(top_vars.items(), key=lambda x: x[1], reverse=True)[:5])


                        if not top_vars:
                            st.warning("No se encontraron variables con buen ajuste para regresión lineal.")
                            continue


                        st.markdown("### Selecciona variable independiente")
                        seleccion = st.selectbox(
                            f"Variable para predecir el precio en {nombre_tab}:",
                            options=list(top_vars.keys()),
                            key=f"{nombre_tab}_selector"
                        )

                        df_valid = df_tipo.dropna(subset=[seleccion])
                        X = df_valid[[seleccion]]
                        y = df_valid["price"]

                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)

                        coef_deter = model.score(X, y)
                        coef_correl = np.sqrt(coef_deter) if coef_deter >= 0 else 0

                    
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown("### Dispersión de datos reales")
                            fig_disp = px.scatter()
                            fig_disp.add_scatter(x=df_valid["room_type"], y=df_valid["price"], mode="markers",
                                                name="room_type vs price", marker=dict(color="#78B6C3"))
                            fig_disp.add_scatter(x=df_valid[seleccion], y=df_valid["price"], mode="markers",
                                                name=f"{seleccion} vs price", marker=dict(color="#3E7D95"))
                            fig_disp.update_layout(
                                title=f"{nombre_tab}: Dispersión de room_type y {seleccion} vs price",
                                xaxis_title="room_type (codificada)",
                                yaxis_title="price"
                            )
                            st.plotly_chart(fig_disp, use_container_width=True)

                            st.markdown("### Gráfico de Regresión")
                            fig = px.scatter(x=df_valid[seleccion], y=y, labels={"x": seleccion, "y": "price"},
                                            title=f"{nombre_tab}: Real vs Predicho")
                            fig.add_scatter(x=df_valid[seleccion], y=y_pred, mode="lines", name="Predicción", line=dict(color="#3E7D95"))
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.markdown("### Métricas del modelo")
                            st.plotly_chart(crear_donut_metrica(model.intercept_, "Intercepto"), use_container_width=True, key=f"{nombre_tab}_intercepto")
                            st.plotly_chart(crear_donut_metrica(coef_deter, "R² (Determinación)"), use_container_width=True, key=f"{nombre_tab}_r2")
                            st.plotly_chart(crear_donut_metrica(coef_correl, "r (Correlación)"), use_container_width=True, key=f"{nombre_tab}_r")

            if vista == "Lineal Múltiple": 
                st.subheader("Regresión Lineal Múltiple")

                # Variables numéricas sin 'price'
                numeric_cols = dQ.select_dtypes(include=['int64', 'float64']).columns.tolist()
                opciones_independientes = [col for col in numeric_cols if col != 'price']

                seleccionadas = st.multiselect("Entrena tu modelo:", options=opciones_independientes)

                if seleccionadas:
                    # Variables
                    Vars_Indep = dQ[seleccionadas].dropna()
                    Var_Dep = dQ.loc[Vars_Indep.index, "price"]

                    # Modelo
                    model = LinearRegression()
                    model.fit(X=Vars_Indep, y=Var_Dep)
                    y_pred = model.predict(X=Vars_Indep)

                    # En esta sección se determinar la mejor variable simple para graficar
                    mejor_var = None
                    mejor_r2 = 0
                    for var in seleccionadas:
                        try:
                            X_temp = Vars_Indep[[var]]
                            model_temp = LinearRegression().fit(X_temp, Var_Dep)
                            r2 = model_temp.score(X_temp, Var_Dep)
                            if r2 > mejor_r2:
                                mejor_r2 = r2
                                mejor_var = var
                        except:
                            continue

                    coef_Deter = model.score(Vars_Indep, Var_Dep)
                    coef_Correl = np.sqrt(coef_Deter) if coef_Deter >= 0 else 0

                    # DataFrame para gráficos
                    f1 = Vars_Indep.copy()
                    f1["price"] = Var_Dep
                    f1["Pred_price"] = y_pred

                    # Layout: gráficos izquierda, métricas derecha
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown("### Gráfico comparativo real vs predicho (una variable)")

                        if mejor_var:
                            x1 = mejor_var
                            x2 = seleccionadas[0] if mejor_var != seleccionadas[0] else (seleccionadas[1] if len(seleccionadas) > 1 else mejor_var)
                            fig1, ax1 = plt.subplots()
                            sns.scatterplot(x=x1, y=x2, color='green', data=f1, label='Real', ax=ax1)
                            sns.scatterplot(x=x1, y=f1["Pred_price"], color='pink', label='Predicho', data=f1, ax=ax1)
                            ax1.set_title("Relación simple entre variables")
                            st.pyplot(fig1)

                        st.markdown("### Gráfico real vs predicho del modelo múltiple")
                        fig2, ax2 = plt.subplots()
                        sns.scatterplot(x="price", y="price", data=f1, label="Real", color="blue", ax=ax2)
                        sns.scatterplot(x="price", y="Pred_price", data=f1, label="Predicho", color="red", ax=ax2)
                        ax2.set_title("Real vs Predicho")
                        st.pyplot(fig2)

                    with col2:
                        st.markdown("### Métricas del modelo")
                        st.plotly_chart(crear_donut_metrica(model.intercept_, "Intercepto"), use_container_width=True, key="intercepto_mult")
                        st.plotly_chart(crear_donut_metrica(coef_Deter, "R² (Determinación)"), use_container_width=True, key="r2_mult")
                        st.plotly_chart(crear_donut_metrica(coef_Correl, "r (Correlación)"), use_container_width=True, key="r_mult")

                else:
                    st.info("Selecciona al menos una variable para entrenar el modelo.")



            if vista == "Logística":
                st.subheader("Regresión Logística")

                # Copia de trabajo del DataFrame
                df_log = dQ.copy()

                # Detectar variables dicotómicas
                dicotomicas = [col for col in df_log.columns if df_log[col].nunique() == 2]
                if "room_type" in df_log.columns:
                    dicotomicas.append("room_type")
                dicotomicas = list(set(dicotomicas))

                # Variables independientes candidatas
                numeric_cols = df_log.select_dtypes(include=['int64', 'float64']).columns.tolist()

                # Selección de variable dependiente
                var_dep = st.selectbox("Selecciona la variable dependiente dicotómica:", options=dicotomicas)

                # Variables independientes 
                opciones_indep = [col for col in numeric_cols if col != var_dep]
                vars_indep = st.multiselect("Selecciona las variables independientes:", options=opciones_indep)

                if var_dep and vars_indep:
                    df_model = df_log.dropna(subset=vars_indep + [var_dep])
                    X = df_model[vars_indep]

                    # Conversión de variable dependiente
                    if var_dep == "room_type":
                        y = df_model["room_type"].apply(lambda x: 1 if x == 1 else 0)  # 1 = Entire
                    else:
                        y = df_model[var_dep]
                        if sorted(y.unique()) != [0, 1]:
                            y = y.map({y.unique()[0]: 0, y.unique()[1]: 1})

                    # División de datos
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # Escalado
                    escalar = StandardScaler()
                    X_train = escalar.fit_transform(X_train)
                    X_test = escalar.transform(X_test)

                    # Modelo
                    modelo = LogisticRegression(max_iter=1000)
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)

                    # Métricas
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    precision = precision_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
                    sensibilidad = recall_score(y_test, y_pred)

                    cm_ordenada = np.array([
                        [cm[1, 1], cm[1, 0]],
                        [cm[0, 1], cm[0, 0]]
                    ])

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("### Matriz de Confusión")
                        fig_cm, ax = plt.subplots()
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm_ordenada)
                        disp.plot(cmap="Purples", ax=ax, colorbar=True)
                        ax.set_title("Matriz de Confusión", fontsize=14)

                        # Etiquetas explicativas
                        ax.text(-0.35, -0.35, "VP\nVerdaderos Positivos", fontsize=8, color="white", weight="bold")
                        ax.text(0.65, -0.35, "FP\nFalsos Positivos", fontsize=8, color="black", weight="bold")
                        ax.text(-0.35, 0.65, "FN\nFalsos Negativos", fontsize=8, color="black", weight="bold")
                        ax.text(0.65, 0.65, "VN\nVerdaderos Negativos", fontsize=8, color="green", weight="bold")

                        st.pyplot(fig_cm)

                    with col2:
                        st.markdown("### Métricas del modelo")
                        st.plotly_chart(crear_donut_metrica(precision, "Precisión"), use_container_width=True, key="log_prec")
                        st.plotly_chart(crear_donut_metrica(accuracy, "Exactitud"), use_container_width=True, key="log_acc")
                        st.plotly_chart(crear_donut_metrica(sensibilidad, "Sensibilidad"), use_container_width=True, key="log_sens")
                else:
                    st.info("Selecciona una variable dependiente dicotómica y al menos una independiente.")

############################################################################################################################################################
    if pais == "Canada" and ciudad == "Victoria": 
        st.title("Victoria, Canada")
        sub_opcion = option_menu(
            menu_title=None,
            options=["DataBase", "Univariado", "Regresiones"],
            icons=["table", "bar-chart-line", "activity"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#DCEAF4"},
                "icon": {"color": "#2C3E50", "font-size": "25px"},
                "nav-link": {
                    "font-size": "16px",
                    "margin": "5px",
                    "--hover-color": "#BEE3F8"
                },
                "nav-link-selected": {"background-color": "#B7CDE2"},
            }
        )

        if sub_opcion == "DataBase": 
            st.title("Información relevante de hospedajes")

            col1, col2, col3 = st.columns(3)

            # Valores por defecto
            precio_min = int(dV["price"].min())
            precio_max = int(dV["price"].max())
            room_types = dV["room_type"].unique().tolist()
            default_columns = ["room_type", "price", "neighbourhood_cleansed"]

            # Estado inicial
            if "precio_filtro" not in st.session_state:
                st.session_state["precio_filtro"] = (precio_min, precio_max)
            if "room_filtro" not in st.session_state:
                st.session_state["room_filtro"] = room_types
            if "cols_filtro" not in st.session_state:
                st.session_state["cols_filtro"] = default_columns

            # Cargamos valores desde el estado
            precio_val = st.session_state["precio_filtro"]
            room_val = st.session_state["room_filtro"]
            cols_val = st.session_state["cols_filtro"]

            with col1:  
                st.markdown("### Filtros de búsqueda")  
            with col2: 
                room_val = st.multiselect("Filtrar por tipo de cuarto", options=room_types, default=room_val)
            with col3: 
                cols_val = st.multiselect("Selecciona columnas a visualizar", options=dV.columns.tolist(), default=cols_val)

            # Guardamos valores nuevamente al estado (para siguiente renderizado)
            st.session_state["precio_filtro"] = precio_val
            st.session_state["room_filtro"] = room_val
            st.session_state["cols_filtro"] = cols_val

            # Aplicar filtros al DataFrame
            dV_filtrado = dV[
                (dV["room_type"].isin(room_val))
            ]

            # Mostrar DataFrame filtrado y resumen
            st.markdown(f"### Datos filtrados ({dV_filtrado.shape[0]} filas)")
            st.dataframe(dV_filtrado[cols_val], use_container_width=True)

            st.markdown("### Resumen estadístico")
            st.dataframe(dV_filtrado.describe(), use_container_width=True)
        
        if sub_opcion == "Univariado": 
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribución Superhosts")
                freq_data = dV['host_is_superhost'].value_counts().reset_index()
                freq_data.columns = ['category', 'count']
                
                figure1 = px.bar(
                    freq_data,
                    x='category',
                    y='count',
                    title="Superhosts (0=No, 1=Sí)",
                    labels={'category': '', 'count': 'Frecuencia'},
                    color='category',
                    color_discrete_map={0: '#6d38aa', 1: '#fbb77c'}
                )
                st.plotly_chart(figure1, use_container_width=True)
                
            with col2:
                st.subheader("Distribución Precios")
                # Solución directa sin usar histograma intermedio
                price_counts = dV['price'].value_counts().sort_index().reset_index()
                price_counts.columns = ['price', 'count']
                
                figure2 = px.line(
                    price_counts,
                    x='price',
                    y='count',
                    title=" ",
                    labels={'price': 'Precio', 'count': 'Frecuencia'}
                )
                st.plotly_chart(figure2, use_container_width=True)

            with col1:
                st.subheader("Distribución de Tipos de Habitación")
                room_counts = dV['room_type'].value_counts().reset_index()
                room_counts.columns = ['room_type', 'count']
                
                figure3 = px.pie(
                    room_counts,
                    names='room_type',
                    values='count',
                    title="Tipo de Habitación:",
                    subtitle="1.-Entire home/apt, 2.-Private room, 3.-Shared room, 4.-Hotel room.",
                )
                st.plotly_chart(figure3, use_container_width=True)

            with  col2:
                # 2. Distribución de Reviews (Scatterplot)
                st.subheader("Puntuaciones de Reviews")
                review_freq = dV['review_scores_rating'].value_counts().reset_index()
                fig2 = px.scatter(
                    review_freq,
                    x='review_scores_rating',
                    y='count',
                    size='count',
                    color='count',
                    title="Frecuencia de Puntuaciones",
                    labels={'review_scores_rating': 'Puntuación', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                    
            #Tabla de frecuencias
            st.subheader("Tablas Resumen de Frecuencias por Variable")
            # Variables categóricas de interés
            variables_frecuencia = ['host_is_superhost', 'room_type', 'bedrooms', 'review_scores_rating']

            for var in variables_frecuencia:
                freq = dV[var].value_counts().reset_index()
                freq.columns = ['Valor', 'Frecuencia']
                freq = freq[freq['Frecuencia'] > 50]
                freq['Porcentaje'] = (freq['Frecuencia'] / freq['Frecuencia'].sum()) * 100
                freq['Variable'] = var

                st.markdown(f"#### Variable: {var}")
                st.dataframe(
                    freq[[ 'Valor', 'Frecuencia', 'Porcentaje']],
                    use_container_width=True,
                    hide_index=True
                )


        if sub_opcion == "Regresiones":
            st.title("Modelados Predictivos")
                
            def crear_donut_metrica(valor, etiqueta, color="#83C5BE"):
                # Si el valor es mayor a 1, lo escalamos para que se vea bien en el donut
                if valor > 1:
                    # Usamos un valor máximo de referencia como el múltiplo de 10 más cercano
                    max_valor = round(valor, -1) if valor < 100 else round(valor, -2)
                    proporcion = min(valor / max_valor, 1)
                else:
                    proporcion = valor  # ya está en escala 0 a 1

                fig = go.Figure(go.Pie(
                    values=[proporcion, 1 - proporcion],
                    labels=["", ""],
                    hole=0.7,
                    marker_colors=[color, "#F0F0F0"],
                    textinfo='none'
                ))
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    annotations=[dict(
                        text=f"<b>{etiqueta}</b><br><span style='font-size:24px'>{valor:.2f}</span>",
                        x=0.5, y=0.5, font_size=20, showarrow=False, align='center'
                    )],
                    width=220, height=220
                )
                return fig

            #heatmap de correlación
            # Limpiar duplicados de columnas, si los hubiera
            dV = dV.loc[:, ~dV.columns.duplicated()]
            numeric_cols = dV.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if 'price' in dV.columns:
                st.subheader("Heatmap de variables con alta correlación con 'price'")
                # Calcular correlación con 'price'
                corr_matrix = dV[numeric_cols].corr()
                correlaciones_con_price = corr_matrix['price'].drop('price').abs()
                
                columnas_filtradas = correlaciones_con_price[correlaciones_con_price >= 0.3].index.tolist()
                columnas_heatmap = columnas_filtradas + ['price']
                
                # Generar heatmap
                fig = px.imshow(
                    dV[columnas_heatmap].corr().abs().round(2),
                    text_auto=True,
                    color_continuous_scale=["#F5F5F5", "#B39CD0", "#6A3FA0"]
                )
                fig.update_layout(width=900, height=500)
                st.plotly_chart(fig, use_container_width=True)

            vista=st.selectbox("Selecciona el tipo de regresión que deseas visualizar", ["Lineal Simple", "Lineal Múltiple", "Logística"])

            if vista == "Lineal Simple":
                st.subheader("Regresión Lineal Simple")

                # Tabs por tipo de cuarto codificado
                tabs = st.tabs(["Entire home/apt", "Private room", "Shared room", "Hotel room"])
                tipos = {
                    "Entire home/apt": 1,
                    "Private room": 2,
                    "Shared room": 3,
                    "Hotel room": 4
                }

                # Generar lista de columnas numéricas limpias
                numeric_cols = dV.select_dtypes(include=['int64', 'float64']).columns.tolist()
                columnas_a_excluir = [
                    "price",
                    "calculated_host_listings_count",
                    "calculated_host_listings_count_entire_homes",
                    "calculated_host_listings_count_private_rooms",
                    "calculated_host_listings_count_shared_rooms"
                ]
                numeric_cols_limpias = [col for col in numeric_cols if col not in columnas_a_excluir]

                for nombre_tab, tipo_valor in zip(tipos.keys(), tabs):
                    with tipo_valor:
                        df_tipo = dV[dV["room_type"] == tipos[nombre_tab]].dropna(subset=["price"])

                        if df_tipo.shape[0] < 2:
                            st.warning("No hay suficientes datos para este tipo de cuarto.")
                            continue

                        # Correlaciones con price
                        # Evaluar R² de regresión lineal simple para cada variable
                        top_vars = {}
                        for col in numeric_cols_limpias:
                            try:
                                df_tmp = df_tipo.dropna(subset=["price", col])
                                if df_tmp.shape[0] < 2:
                                    continue
                                X_tmp = df_tmp[[col]]
                                y_tmp = df_tmp["price"]
                                model_tmp = LinearRegression().fit(X_tmp, y_tmp)
                                r2 = model_tmp.score(X_tmp, y_tmp)
                                if r2 > 0.01:  # umbral bajo para evitar ruido
                                    top_vars[col] = r2
                            except:
                                pass

                        # Ordenar por R² descendente y quedarte con top 5
                        top_vars = dict(sorted(top_vars.items(), key=lambda x: x[1], reverse=True)[:5])


                        if not top_vars:
                            st.warning("No se encontraron variables con buen ajuste para regresión lineal.")
                            continue


                        st.markdown("### Selecciona variable independiente")
                        seleccion = st.selectbox(
                            f"Variable para predecir el precio en {nombre_tab}:",
                            options=list(top_vars.keys()),
                            key=f"{nombre_tab}_selector"
                        )

                        df_valid = df_tipo.dropna(subset=[seleccion])
                        X = df_valid[[seleccion]]
                        y = df_valid["price"]

                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)

                        coef_deter = model.score(X, y)
                        coef_correl = np.sqrt(coef_deter) if coef_deter >= 0 else 0

                    
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown("### Dispersión de datos reales")
                            fig_disp = px.scatter()
                            fig_disp.add_scatter(x=df_valid["room_type"], y=df_valid["price"], mode="markers",
                                                name="room_type vs price", marker=dict(color="#78B6C3"))
                            fig_disp.add_scatter(x=df_valid[seleccion], y=df_valid["price"], mode="markers",
                                                name=f"{seleccion} vs price", marker=dict(color="#3E7D95"))
                            fig_disp.update_layout(
                                title=f"{nombre_tab}: Dispersión de room_type y {seleccion} vs price",
                                xaxis_title="room_type (codificada)",
                                yaxis_title="price"
                            )
                            st.plotly_chart(fig_disp, use_container_width=True)

                            st.markdown("### Gráfico de Regresión")
                            fig = px.scatter(x=df_valid[seleccion], y=y, labels={"x": seleccion, "y": "price"},
                                            title=f"{nombre_tab}: Real vs Predicho")
                            fig.add_scatter(x=df_valid[seleccion], y=y_pred, mode="lines", name="Predicción", line=dict(color="#3E7D95"))
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.markdown("### Métricas del modelo")
                            st.plotly_chart(crear_donut_metrica(model.intercept_, "Intercepto"), use_container_width=True, key=f"{nombre_tab}_intercepto")
                            st.plotly_chart(crear_donut_metrica(coef_deter, "R² (Determinación)"), use_container_width=True, key=f"{nombre_tab}_r2")
                            st.plotly_chart(crear_donut_metrica(coef_correl, "r (Correlación)"), use_container_width=True, key=f"{nombre_tab}_r")

            if vista == "Lineal Múltiple": 
                st.subheader("Regresión Lineal Múltiple")

                # Variables numéricas sin 'price'
                numeric_cols = dV.select_dtypes(include=['int64', 'float64']).columns.tolist()
                opciones_independientes = [col for col in numeric_cols if col != 'price']

                seleccionadas = st.multiselect("Entrena tu modelo:", options=opciones_independientes)

                if seleccionadas:
                    # Variables
                    Vars_Indep = dV[seleccionadas].dropna()
                    Var_Dep = dV.loc[Vars_Indep.index, "price"]

                    # Modelo
                    model = LinearRegression()
                    model.fit(X=Vars_Indep, y=Var_Dep)
                    y_pred = model.predict(X=Vars_Indep)

                    # En esta sección se determinar la mejor variable simple para graficar
                    mejor_var = None
                    mejor_r2 = 0
                    for var in seleccionadas:
                        try:
                            X_temp = Vars_Indep[[var]]
                            model_temp = LinearRegression().fit(X_temp, Var_Dep)
                            r2 = model_temp.score(X_temp, Var_Dep)
                            if r2 > mejor_r2:
                                mejor_r2 = r2
                                mejor_var = var
                        except:
                            continue

                    coef_Deter = model.score(Vars_Indep, Var_Dep)
                    coef_Correl = np.sqrt(coef_Deter) if coef_Deter >= 0 else 0

                    # DataFrame para gráficos
                    f1 = Vars_Indep.copy()
                    f1["price"] = Var_Dep
                    f1["Pred_price"] = y_pred

                    # Layout: gráficos izquierda, métricas derecha
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown("### Gráfico comparativo real vs predicho (una variable)")

                        if mejor_var:
                            x1 = mejor_var
                            x2 = seleccionadas[0] if mejor_var != seleccionadas[0] else (seleccionadas[1] if len(seleccionadas) > 1 else mejor_var)
                            fig1, ax1 = plt.subplots()
                            sns.scatterplot(x=x1, y=x2, color='green', data=f1, label='Real', ax=ax1)
                            sns.scatterplot(x=x1, y=f1["Pred_price"], color='pink', label='Predicho', data=f1, ax=ax1)
                            ax1.set_title("Relación simple entre variables")
                            st.pyplot(fig1)

                        st.markdown("### Gráfico real vs predicho del modelo múltiple")
                        fig2, ax2 = plt.subplots()
                        sns.scatterplot(x="price", y="price", data=f1, label="Real", color="blue", ax=ax2)
                        sns.scatterplot(x="price", y="Pred_price", data=f1, label="Predicho", color="red", ax=ax2)
                        ax2.set_title("Real vs Predicho")
                        st.pyplot(fig2)

                    with col2:
                        st.markdown("### Métricas del modelo")
                        st.plotly_chart(crear_donut_metrica(model.intercept_, "Intercepto"), use_container_width=True, key="intercepto_mult")
                        st.plotly_chart(crear_donut_metrica(coef_Deter, "R² (Determinación)"), use_container_width=True, key="r2_mult")
                        st.plotly_chart(crear_donut_metrica(coef_Correl, "r (Correlación)"), use_container_width=True, key="r_mult")

                else:
                    st.info("Selecciona al menos una variable para entrenar el modelo.")



            if vista == "Logística":
                st.subheader("Regresión Logística")

                # Copia de trabajo del DataFrame
                df_log = dV.copy()

                # Detectar variables dicotómicas
                dicotomicas = [col for col in df_log.columns if df_log[col].nunique() == 2]
                if "room_type" in df_log.columns:
                    dicotomicas.append("room_type")
                dicotomicas = list(set(dicotomicas))

                # Variables independientes candidatas
                numeric_cols = df_log.select_dtypes(include=['int64', 'float64']).columns.tolist()

                # Selección de variable dependiente
                var_dep = st.selectbox("Selecciona la variable dependiente dicotómica:", options=dicotomicas)

                # Variables independientes 
                opciones_indep = [col for col in numeric_cols if col != var_dep]
                vars_indep = st.multiselect("Selecciona las variables independientes:", options=opciones_indep)

                if var_dep and vars_indep:
                    df_model = df_log.dropna(subset=vars_indep + [var_dep])
                    X = df_model[vars_indep]

                    # Conversión de variable dependiente
                    if var_dep == "room_type":
                        y = df_model["room_type"].apply(lambda x: 1 if x == 1 else 0)  # 1 = Entire
                    else:
                        y = df_model[var_dep]
                        if sorted(y.unique()) != [0, 1]:
                            y = y.map({y.unique()[0]: 0, y.unique()[1]: 1})

                    # División de datos
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # Escalado
                    escalar = StandardScaler()
                    X_train = escalar.fit_transform(X_train)
                    X_test = escalar.transform(X_test)

                    # Modelo
                    modelo = LogisticRegression(max_iter=1000)
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)

                    # Métricas
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    precision = precision_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
                    sensibilidad = recall_score(y_test, y_pred)

                    cm_ordenada = np.array([
                        [cm[1, 1], cm[1, 0]],
                        [cm[0, 1], cm[0, 0]]
                    ])

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("### Matriz de Confusión")
                        fig_cm, ax = plt.subplots()
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm_ordenada)
                        disp.plot(cmap="Purples", ax=ax, colorbar=True)
                        ax.set_title("Matriz de Confusión", fontsize=14)

                        # Etiquetas explicativas
                        ax.text(-0.35, -0.35, "VP\nVerdaderos Positivos", fontsize=8, color="white", weight="bold")
                        ax.text(0.65, -0.35, "FP\nFalsos Positivos", fontsize=8, color="black", weight="bold")
                        ax.text(-0.35, 0.65, "FN\nFalsos Negativos", fontsize=8, color="black", weight="bold")
                        ax.text(0.65, 0.65, "VN\nVerdaderos Negativos", fontsize=8, color="green", weight="bold")

                        st.pyplot(fig_cm)

                    with col2:
                        st.markdown("### Métricas del modelo")
                        st.plotly_chart(crear_donut_metrica(precision, "Precisión"), use_container_width=True, key="log_prec")
                        st.plotly_chart(crear_donut_metrica(accuracy, "Exactitud"), use_container_width=True, key="log_acc")
                        st.plotly_chart(crear_donut_metrica(sensibilidad, "Sensibilidad"), use_container_width=True, key="log_sens")
                else:
                    st.info("Selecciona una variable dependiente dicotómica y al menos una independiente.")

###########################################################################################################################################################    
    if pais == "Italia" and ciudad == "Malta":
        st.title("Malta Italia")
        sub_opcion = option_menu(
            menu_title=None,
            options=["DataBase", "Univariado", "Regresiones"],
            icons=["table", "bar-chart-line", "activity"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#DCEAF4"},
                "icon": {"color": "#2C3E50", "font-size": "25px"},
                "nav-link": {
                    "font-size": "16px",
                    "margin": "5px",
                    "--hover-color": "#BEE3F8"
                },
                "nav-link-selected": {"background-color": "#B7CDE2"},
            }
        )

        if sub_opcion == "DataBase": 
            st.title("Información relevante de hospedajes")

            col1, col2, col3 = st.columns(3)

            # Valores por defecto
            precio_min = int(dM["price"].min())
            precio_max = int(dM["price"].max())
            room_types = dM["room_type"].unique().tolist()
            default_columns = [col for col in ["room_type", "price"]]

            # Estado inicial
            if "precio_filtro" not in st.session_state:
                st.session_state["precio_filtro"] = (precio_min, precio_max)
            if "room_filtro" not in st.session_state:
                st.session_state["room_filtro"] = room_types
            if "cols_filtro" not in st.session_state:
                st.session_state["cols_filtro"] = default_columns

            # Cargamos valores desde el estado
            precio_val = st.session_state["precio_filtro"]
            room_val = st.session_state["room_filtro"]
            cols_val = st.session_state["cols_filtro"]

            with col1:  
                st.markdown("### Filtros de búsqueda")  
            with col2: 
                room_val = st.multiselect("Filtrar por tipo de cuarto", options=room_types, default=room_val)
            with col3: 
                opciones_cols = dM.columns.tolist()
                cols_val = [col for col in cols_val if col in opciones_cols]  # Asegura que sean válidas
                cols_val = st.multiselect("Selecciona columnas a visualizar", options=opciones_cols, default=cols_val)


            # Guardamos valores nuevamente al estado (para siguiente renderizado)
            st.session_state["precio_filtro"] = precio_val
            st.session_state["room_filtro"] = room_val
            st.session_state["cols_filtro"] = cols_val

            # Aplicar filtros al DataFrame
            dM_filtrado = dM[
                (dM["room_type"].isin(room_val))
            ]

            # Mostrar DataFrame filtrado y resumen
            st.markdown(f"### Datos filtrados ({dM_filtrado.shape[0]} filas)")
            st.dataframe(dM_filtrado[cols_val], use_container_width=True)

            st.markdown("### Resumen estadístico")
            st.dataframe(dM_filtrado.describe(), use_container_width=True)
        
        if sub_opcion == "Univariado": 
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribución Superhosts")
                freq_data = dM['host_is_superhost'].value_counts().reset_index()
                freq_data.columns = ['category', 'count']
                
                figure1 = px.bar(
                    freq_data,
                    x='category',
                    y='count',
                    title="Superhosts (0=No, 1=Sí)",
                    labels={'category': '', 'count': 'Frecuencia'},
                    color='category',
                    color_discrete_map={0: '#6d38aa', 1: '#fbb77c'}
                )
                st.plotly_chart(figure1, use_container_width=True)
                
            with col2:
                st.subheader("Distribución Precios")
                # Solución directa sin usar histograma intermedio
                price_counts = dM['price'].value_counts().sort_index().reset_index()
                price_counts.columns = ['price', 'count']
                
                figure2 = px.line(
                    price_counts,
                    x='price',
                    y='count',
                    title=" ",
                    labels={'price': 'Precio', 'count': 'Frecuencia'}
                )
                st.plotly_chart(figure2, use_container_width=True)

            with col1:
                st.subheader("Distribución de Tipos de Habitación")
                room_counts = dM['room_type'].value_counts().reset_index()
                room_counts.columns = ['room_type', 'count']
                
                figure3 = px.pie(
                    room_counts,
                    names='room_type',
                    values='count',
                    title="Tipo de Habitación:",
                    subtitle="1.-Entire home/apt, 2.-Private room, 3.-Shared room, 4.-Hotel room.",
                )
                st.plotly_chart(figure3, use_container_width=True)

            with  col2:
                # 2. Distribución de Reviews (Scatterplot)
                st.subheader("Puntuaciones de Reviews")
                review_freq = dM['review_scores_rating'].value_counts().reset_index()
                fig2 = px.scatter(
                    review_freq,
                    x='review_scores_rating',
                    y='count',
                    size='count',
                    color='count',
                    title="Frecuencia de Puntuaciones",
                    labels={'review_scores_rating': 'Puntuación', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                    
            #Tabla de frecuencias
            st.subheader("Tablas Resumen de Frecuencias por Variable")
            # Variables categóricas de interés
            variables_frecuencia = ['host_is_superhost', 'room_type', 'bedrooms', 'review_scores_rating']

            for var in variables_frecuencia:
                freq = dM[var].value_counts().reset_index()
                freq.columns = ['Valor', 'Frecuencia']
                freq = freq[freq['Frecuencia'] > 50]
                freq['Porcentaje'] = (freq['Frecuencia'] / freq['Frecuencia'].sum()) * 100
                freq['Variable'] = var

                st.markdown(f"#### Variable: {var}")
                st.dataframe(
                    freq[[ 'Valor', 'Frecuencia', 'Porcentaje']],
                    use_container_width=True,
                    hide_index=True
                )


        if sub_opcion == "Regresiones":
            st.title("Modelados Predictivos")
                
            def crear_donut_metrica(valor, etiqueta, color="#83C5BE"):
                # Si el valor es mayor a 1, lo escalamos para que se vea bien en el donut
                if valor > 1:
                    # Usamos un valor máximo de referencia como el múltiplo de 10 más cercano
                    max_valor = round(valor, -1) if valor < 100 else round(valor, -2)
                    proporcion = min(valor / max_valor, 1)
                else:
                    proporcion = valor  # ya está en escala 0 a 1

                fig = go.Figure(go.Pie(
                    values=[proporcion, 1 - proporcion],
                    labels=["", ""],
                    hole=0.7,
                    marker_colors=[color, "#F0F0F0"],
                    textinfo='none'
                ))
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    annotations=[dict(
                        text=f"<b>{etiqueta}</b><br><span style='font-size:24px'>{valor:.2f}</span>",
                        x=0.5, y=0.5, font_size=20, showarrow=False, align='center'
                    )],
                    width=220, height=220
                )
                return fig

            #heatmap de correlación
            # Limpiar duplicados de columnas, si los hubiera
            dM = dM.loc[:, ~dM.columns.duplicated()]
            numeric_cols = dM.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if 'price' in dM.columns:
                st.subheader("Heatmap de variables con alta correlación con 'price'")
                # Calcular correlación con 'price'
                corr_matrix = dM[numeric_cols].corr()
                correlaciones_con_price = corr_matrix['price'].drop('price').abs()
                
                columnas_filtradas = correlaciones_con_price[correlaciones_con_price >= 0.3].index.tolist()
                columnas_heatmap = columnas_filtradas + ['price']
                
                # Generar heatmap
                fig = px.imshow(
                    dM[columnas_heatmap].corr().abs().round(2),
                    text_auto=True,
                    color_continuous_scale=["#F5F5F5", "#B39CD0", "#6A3FA0"]
                )
                fig.update_layout(width=900, height=500)
                st.plotly_chart(fig, use_container_width=True)

            vista=st.selectbox("Selecciona el tipo de regresión que deseas visualizar", ["Lineal Simple", "Lineal Múltiple", "Logística"])

            if vista == "Lineal Simple":
                st.subheader("Regresión Lineal Simple")

                # Tabs por tipo de cuarto codificado
                tabs = st.tabs(["Entire home/apt", "Private room", "Shared room", "Hotel room"])
                tipos = {
                    "Entire home/apt": 1,
                    "Private room": 2,
                    "Shared room": 3,
                    "Hotel room": 4
                }

                # Generar lista de columnas numéricas limpias
                numeric_cols = dM.select_dtypes(include=['int64', 'float64']).columns.tolist()
                columnas_a_excluir = [
                    "price",
                    "calculated_host_listings_count",
                    "calculated_host_listings_count_entire_homes",
                    "calculated_host_listings_count_private_rooms",
                    "calculated_host_listings_count_shared_rooms"
                ]
                numeric_cols_limpias = [col for col in numeric_cols if col not in columnas_a_excluir]

                for nombre_tab, tipo_valor in zip(tipos.keys(), tabs):
                    with tipo_valor:
                        df_tipo = dM[dM["room_type"] == tipos[nombre_tab]].dropna(subset=["price"])

                        if df_tipo.shape[0] < 2:
                            st.warning("No hay suficientes datos para este tipo de cuarto.")
                            continue

                        # Correlaciones con price
                        # Evaluar R² de regresión lineal simple para cada variable
                        top_vars = {}
                        for col in numeric_cols_limpias:
                            try:
                                df_tmp = df_tipo.dropna(subset=["price", col])
                                if df_tmp.shape[0] < 2:
                                    continue
                                X_tmp = df_tmp[[col]]
                                y_tmp = df_tmp["price"]
                                model_tmp = LinearRegression().fit(X_tmp, y_tmp)
                                r2 = model_tmp.score(X_tmp, y_tmp)
                                if r2 > 0.01:  # umbral bajo para evitar ruido
                                    top_vars[col] = r2
                            except:
                                pass

                        # Ordenar por R² descendente y quedarte con top 5
                        top_vars = dict(sorted(top_vars.items(), key=lambda x: x[1], reverse=True)[:5])


                        if not top_vars:
                            st.warning("No se encontraron variables con buen ajuste para regresión lineal.")
                            continue


                        st.markdown("### Selecciona variable independiente")
                        seleccion = st.selectbox(
                            f"Variable para predecir el precio en {nombre_tab}:",
                            options=list(top_vars.keys()),
                            key=f"{nombre_tab}_selector"
                        )

                        df_valid = df_tipo.dropna(subset=[seleccion])
                        X = df_valid[[seleccion]]
                        y = df_valid["price"]

                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)

                        coef_deter = model.score(X, y)
                        coef_correl = np.sqrt(coef_deter) if coef_deter >= 0 else 0

                    
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown("### Dispersión de datos reales")
                            fig_disp = px.scatter()
                            fig_disp.add_scatter(x=df_valid["room_type"], y=df_valid["price"], mode="markers",
                                                name="room_type vs price", marker=dict(color="#78B6C3"))
                            fig_disp.add_scatter(x=df_valid[seleccion], y=df_valid["price"], mode="markers",
                                                name=f"{seleccion} vs price", marker=dict(color="#3E7D95"))
                            fig_disp.update_layout(
                                title=f"{nombre_tab}: Dispersión de room_type y {seleccion} vs price",
                                xaxis_title="room_type (codificada)",
                                yaxis_title="price"
                            )
                            st.plotly_chart(fig_disp, use_container_width=True)

                            st.markdown("### Gráfico de Regresión")
                            fig = px.scatter(x=df_valid[seleccion], y=y, labels={"x": seleccion, "y": "price"},
                                            title=f"{nombre_tab}: Real vs Predicho")
                            fig.add_scatter(x=df_valid[seleccion], y=y_pred, mode="lines", name="Predicción", line=dict(color="#3E7D95"))
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.markdown("### Métricas del modelo")
                            st.plotly_chart(crear_donut_metrica(model.intercept_, "Intercepto"), use_container_width=True, key=f"{nombre_tab}_intercepto")
                            st.plotly_chart(crear_donut_metrica(coef_deter, "R² (Determinación)"), use_container_width=True, key=f"{nombre_tab}_r2")
                            st.plotly_chart(crear_donut_metrica(coef_correl, "r (Correlación)"), use_container_width=True, key=f"{nombre_tab}_r")

            if vista == "Lineal Múltiple": 
                st.subheader("Regresión Lineal Múltiple")

                # Variables numéricas sin 'price'
                numeric_cols = dM.select_dtypes(include=['int64', 'float64']).columns.tolist()
                opciones_independientes = [col for col in numeric_cols if col != 'price']

                seleccionadas = st.multiselect("Entrena tu modelo:", options=opciones_independientes)

                if seleccionadas:
                    # Variables
                    Vars_Indep = dM[seleccionadas].dropna()
                    Var_Dep = dM.loc[Vars_Indep.index, "price"]

                    # Modelo
                    model = LinearRegression()
                    model.fit(X=Vars_Indep, y=Var_Dep)
                    y_pred = model.predict(X=Vars_Indep)

                    # En esta sección se determinar la mejor variable simple para graficar
                    mejor_var = None
                    mejor_r2 = 0
                    for var in seleccionadas:
                        try:
                            X_temp = Vars_Indep[[var]]
                            model_temp = LinearRegression().fit(X_temp, Var_Dep)
                            r2 = model_temp.score(X_temp, Var_Dep)
                            if r2 > mejor_r2:
                                mejor_r2 = r2
                                mejor_var = var
                        except:
                            continue

                    coef_Deter = model.score(Vars_Indep, Var_Dep)
                    coef_Correl = np.sqrt(coef_Deter) if coef_Deter >= 0 else 0

                    # DataFrame para gráficos
                    f1 = Vars_Indep.copy()
                    f1["price"] = Var_Dep
                    f1["Pred_price"] = y_pred

                    # Layout: gráficos izquierda, métricas derecha
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown("### Gráfico comparativo real vs predicho (una variable)")

                        if mejor_var:
                            x1 = mejor_var
                            x2 = seleccionadas[0] if mejor_var != seleccionadas[0] else (seleccionadas[1] if len(seleccionadas) > 1 else mejor_var)
                            fig1, ax1 = plt.subplots()
                            sns.scatterplot(x=x1, y=x2, color='green', data=f1, label='Real', ax=ax1)
                            sns.scatterplot(x=x1, y=f1["Pred_price"], color='pink', label='Predicho', data=f1, ax=ax1)
                            ax1.set_title("Relación simple entre variables")
                            st.pyplot(fig1)

                        st.markdown("### Gráfico real vs predicho del modelo múltiple")
                        fig2, ax2 = plt.subplots()
                        sns.scatterplot(x="price", y="price", data=f1, label="Real", color="blue", ax=ax2)
                        sns.scatterplot(x="price", y="Pred_price", data=f1, label="Predicho", color="red", ax=ax2)
                        ax2.set_title("Real vs Predicho")
                        st.pyplot(fig2)

                    with col2:
                        st.markdown("### Métricas del modelo")
                        st.plotly_chart(crear_donut_metrica(model.intercept_, "Intercepto"), use_container_width=True, key="intercepto_mult")
                        st.plotly_chart(crear_donut_metrica(coef_Deter, "R² (Determinación)"), use_container_width=True, key="r2_mult")
                        st.plotly_chart(crear_donut_metrica(coef_Correl, "r (Correlación)"), use_container_width=True, key="r_mult")

                else:
                    st.info("Selecciona al menos una variable para entrenar el modelo.")



            if vista == "Logística":
                st.subheader("Regresión Logística")

                # Copia de trabajo del DataFrame
                df_log = dM.copy()

                # Detectar variables dicotómicas
                dicotomicas = [col for col in df_log.columns if df_log[col].nunique() == 2]
                if "room_type" in df_log.columns:
                    dicotomicas.append("room_type")
                dicotomicas = list(set(dicotomicas))

                # Variables independientes candidatas
                numeric_cols = df_log.select_dtypes(include=['int64', 'float64']).columns.tolist()

                # Selección de variable dependiente
                var_dep = st.selectbox("Selecciona la variable dependiente dicotómica:", options=dicotomicas)

                # Variables independientes 
                opciones_indep = [col for col in numeric_cols if col != var_dep]
                vars_indep = st.multiselect("Selecciona las variables independientes:", options=opciones_indep)

                if var_dep and vars_indep:
                    df_model = df_log.dropna(subset=vars_indep + [var_dep])
                    X = df_model[vars_indep]

                    # Conversión de variable dependiente
                    if var_dep == "room_type":
                        y = df_model["room_type"].apply(lambda x: 1 if x == 1 else 0)  # 1 = Entire
                    else:
                        y = df_model[var_dep]
                        if sorted(y.unique()) != [0, 1]:
                            y = y.map({y.unique()[0]: 0, y.unique()[1]: 1})

                    # División de datos
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # Escalado
                    escalar = StandardScaler()
                    X_train = escalar.fit_transform(X_train)
                    X_test = escalar.transform(X_test)

                    # Modelo
                    modelo = LogisticRegression(max_iter=1000)
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)

                    # Métricas
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    precision = precision_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
                    sensibilidad = recall_score(y_test, y_pred)

                    cm_ordenada = np.array([
                        [cm[1, 1], cm[1, 0]],
                        [cm[0, 1], cm[0, 0]]
                    ])

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("### Matriz de Confusión")
                        fig_cm, ax = plt.subplots()
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm_ordenada)
                        disp.plot(cmap="Purples", ax=ax, colorbar=True)
                        ax.set_title("Matriz de Confusión", fontsize=14)

                        # Etiquetas explicativas
                        ax.text(-0.35, -0.35, "VP\nVerdaderos Positivos", fontsize=8, color="white", weight="bold")
                        ax.text(0.65, -0.35, "FP\nFalsos Positivos", fontsize=8, color="black", weight="bold")
                        ax.text(-0.35, 0.65, "FN\nFalsos Negativos", fontsize=8, color="black", weight="bold")
                        ax.text(0.65, 0.65, "VN\nVerdaderos Negativos", fontsize=8, color="green", weight="bold")

                        st.pyplot(fig_cm)

                    with col2:
                        st.markdown("### Métricas del modelo")
                        st.plotly_chart(crear_donut_metrica(precision, "Precisión"), use_container_width=True, key="log_prec")
                        st.plotly_chart(crear_donut_metrica(accuracy, "Exactitud"), use_container_width=True, key="log_acc")
                        st.plotly_chart(crear_donut_metrica(sensibilidad, "Sensibilidad"), use_container_width=True, key="log_sens")
                else:
                    st.info("Selecciona una variable dependiente dicotómica y al menos una independiente.")

if pagina == "Comparacion": 
    sub_opcion = option_menu(
        menu_title=None,
        options=["Univariado", "Regresiones"],
        icons=["bar-chart-line", "activity"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#DCEAF4"},
            "icon": {"color": "#2C3E50", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "margin": "5px",
                "--hover-color": "#BEE3F8"
            },
            "nav-link-selected": {"background-color": "#B7CDE2"},
        }
    )

    if sub_opcion == "Univariado":
        grafico = option_menu(
            menu_title=None,
            options=["Pieplot", "Barplot", "Lineplot", "Scatterplot"],
            icons=["pie-chart", "bar-chart", "graph-up", "graph-up"],
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#EAF2F8"},
                "icon": {"color": "#2C3E50", "font-size": "22px"},
                "nav-link": {
                    "font-size": "15px",
                    "margin": "4px",
                    "--hover-color": "#D6EAF8"
                },
                "nav-link-selected": {"background-color": "#AED6F1"},
            }
        )

        if grafico == "Pieplot": 
            st.subheader("Distribución Superhost")
            col1, col2 = st.columns(2)

            with col1: 
                st.subheader("México")
                room_counts = dMx['host_is_superhost'].value_counts().reset_index()
                room_counts.columns = ['host_is_superhost', 'count']
                
                figure3 = px.pie(
                    room_counts,
                    names='host_is_superhost',
                    values='count',
                    title="host_is_superhost:",
                )
                st.plotly_chart(figure3, use_container_width=True)

                st.subheader("Malta")
                room_counts = dM['host_is_superhost'].value_counts().reset_index()
                room_counts.columns = ['host_is_superhost', 'count']
                
                figure3 = px.pie(
                    room_counts,
                    names='host_is_superhost',
                    values='count',
                    title="host_is_superhost:",
                )
                st.plotly_chart(figure3, use_container_width=True)
            
            with col2: 
                st.subheader("Quebec")
                room_counts = dQ['host_is_superhost'].value_counts().reset_index()
                room_counts.columns = ['host_is_superhost', 'count']
                
                figure3 = px.pie(
                    room_counts,
                    names='host_is_superhost',
                    values='count',
                    title="host_is_superhost:",
                )
                st.plotly_chart(figure3, use_container_width=True)

                st.subheader("Victoria")
                room_counts = dV['host_is_superhost'].value_counts().reset_index()
                room_counts.columns = ['host_is_superhost', 'count']
                
                figure3 = px.pie(
                    room_counts,
                    names='host_is_superhost',
                    values='count',
                    title="host_is_superhost:",
                )
                st.plotly_chart(figure3, use_container_width=True)
            
        if grafico == "Barplot":
            st.subheader("Distribución Habitaciones")
            col1, col2 = st.columns(2)

            with col1: 
                st.subheader("México")
                freq_data = dMx['room_type'].value_counts().reset_index()
                freq_data.columns = ['category', 'count']
                
                figure1 = px.bar(
                    freq_data,
                    x='category',
                    y='count',
                    labels={'category': '', 'count': 'Frecuencia'},
                    color='category',
                    #color_discrete_map={0: '#6d38aa', 1: '#fbb77c'}
                )
                st.plotly_chart(figure1, use_container_width=True)

                st.subheader("Malta")
                freq_data = dM['room_type'].value_counts().reset_index()
                freq_data.columns = ['category', 'count']
                
                figure1 = px.bar(
                    freq_data,
                    x='category',
                    y='count',
                    labels={'category': '', 'count': 'Frecuencia'},
                    color='category',
                    #color_discrete_map={0: '#6d38aa', 1: '#fbb77c'}
                )
                st.plotly_chart(figure1, use_container_width=True)
            
            with col2:
                st.subheader("Quebec")
                freq_data = dQ['room_type'].value_counts().reset_index()
                freq_data.columns = ['category', 'count']
                
                figure1 = px.bar(
                    freq_data,
                    x='category',
                    y='count',
                    labels={'category': '', 'count': 'Frecuencia'},
                    color='category',
                    #color_discrete_map={0: '#6d38aa', 1: '#fbb77c'}
                )
                st.plotly_chart(figure1, use_container_width=True)
                
                st.subheader("Victoria")
                freq_data = dV['room_type'].value_counts().reset_index()
                freq_data.columns = ['category', 'count']
                
                figure1 = px.bar(
                    freq_data,
                    x='category',
                    y='count',
                    labels={'category': '', 'count': 'Frecuencia'},
                    color='category',
                    #color_discrete_map={1: '#6d38aa', 2: '#fbb77c'}
                )
                st.plotly_chart(figure1, use_container_width=True)

        if grafico == "Lineplot": 
            st.title("Distribucion de precios")

            col1, col2 = st.columns(2)

            with col1: 
                st.subheader("México")
                # Solución directa sin usar histograma intermedio
                price_counts = dMx['price'].value_counts().sort_index().reset_index()
                price_counts.columns = ['price', 'count']
                
                figure2 = px.line(
                    price_counts,
                    x='price',
                    y='count',
                    title=" ",
                    labels={'price': 'Precio', 'count': 'Frecuencia'}
                )
                st.plotly_chart(figure2, use_container_width=True)

                st.subheader("Malta")
                # Solución directa sin usar histograma intermedio
                price_counts = dM['price'].value_counts().sort_index().reset_index()
                price_counts.columns = ['price', 'count']
                
                figure2 = px.line(
                    price_counts,
                    x='price',
                    y='count',
                    title=" ",
                    labels={'price': 'Precio', 'count': 'Frecuencia'}
                )
                st.plotly_chart(figure2, use_container_width=True)
            
            with col2: 
                st.subheader("Quebec")
                # Solución directa sin usar histograma intermedio
                price_counts = dQ['price'].value_counts().sort_index().reset_index()
                price_counts.columns = ['price', 'count']
                
                figure2 = px.line(
                    price_counts,
                    x='price',
                    y='count',
                    title=" ",
                    labels={'price': 'Precio', 'count': 'Frecuencia'}
                )
                st.plotly_chart(figure2, use_container_width=True)

                st.subheader("Victoria")
                # Solución directa sin usar histograma intermedio
                price_counts = dV['price'].value_counts().sort_index().reset_index()
                price_counts.columns = ['price', 'count']
                
                figure2 = px.line(
                    price_counts,
                    x='price',
                    y='count',
                    title=" ",
                    labels={'price': 'Precio', 'count': 'Frecuencia'}
                )
                st.plotly_chart(figure2, use_container_width=True)
            
        if grafico == "Scatterplot": 
            st.title("Puntuación de reviews")
            col1, col2 = st.columns(2)

            with col1: 
                # 2. Distribución de Reviews (Scatterplot)
                st.subheader("México")
                review_freq = dMx['review_scores_rating'].value_counts().reset_index()
                fig2 = px.scatter(
                    review_freq,
                    x='review_scores_rating',
                    y='count',
                    size='count',
                    color='count',
                    title="Frecuencia de Puntuaciones",
                    labels={'review_scores_rating': 'Puntuación', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Malta")
                review_freq = dM['review_scores_rating'].value_counts().reset_index()
                fig2 = px.scatter(
                    review_freq,
                    x='review_scores_rating',
                    y='count',
                    size='count',
                    color='count',
                    title="Frecuencia de Puntuaciones",
                    labels={'review_scores_rating': 'Puntuación', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            with col2: 
                st.subheader("Quebec")
                review_freq = dQ['review_scores_rating'].value_counts().reset_index()
                fig2 = px.scatter(
                    review_freq,
                    x='review_scores_rating',
                    y='count',
                    size='count',
                    color='count',
                    title="Frecuencia de Puntuaciones",
                    labels={'review_scores_rating': 'Puntuación', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("Victoria")
                review_freq = dV['review_scores_rating'].value_counts().reset_index()
                fig2 = px.scatter(
                    review_freq,
                    x='review_scores_rating',
                    y='count',
                    size='count',
                    color='count',
                    title="Frecuencia de Puntuaciones",
                    labels={'review_scores_rating': 'Puntuación', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig2, use_container_width=True)


    elif sub_opcion == "Regresiones":
        tipo_regresion = option_menu(
            menu_title=None,
            options=["Simple", "Múltiple", "Logística"],
            icons=["activity", "sliders", "graph-up"],
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#EAF2F8"},
                "icon": {"color": "#2C3E50", "font-size": "22px"},
                "nav-link": {
                    "font-size": "15px",
                    "margin": "4px",
                    "--hover-color": "#D6EAF8"
                },
                "nav-link-selected": {"background-color": "#AED6F1"},
            }
        )

        if tipo_regresion == "Simple": 
            #Entire home/apt
            MxE= dMx[(dMx["room_type"] == 1)]
            ME= dM[(dM["room_type"] == 1)]
            QE= dQ[(dQ["room_type"] == 1)]
            VE= dV[(dV["room_type"] == 1)]

            st.title("Regresión Lineal Entire Home/APT")
            col1, col2=  st.columns(2)

            with col1:
                st.subheader("México")
                default_x = "bathrooms" if "accommodates" in numeric_cols2 else numeric_cols[0]
                default_y = "accommodates" if "bathrooms" in numeric_cols2 else numeric_cols[0]

                X = MxE[[default_x]]
                y = MxE[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                
                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)

                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

                st.subheader("Malta")
                default_x = "bathrooms" if "accommodates" in numeric_cols2 else numeric_cols[0]
                default_y = "accommodates" if "bathrooms" in numeric_cols2 else numeric_cols[0]

                X = ME[[default_x]]
                y = ME[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                
                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)

                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")
            
            with col2:
                st.subheader("Quebec")
                default_x = "price" if "accommodates" in numeric_cols2 else numeric_cols[0]
                default_y = "accommodates" if "price" in numeric_cols2 else numeric_cols[0]

                X = QE[[default_x]]
                y = QE[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                
                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

                st.subheader("Victoria")
                default_x = "host_response_rate" if "host_is_superhost" in numeric_cols2 else numeric_cols[0]
                default_y = "host_is_superhost" if "host_response_rate" in numeric_cols2 else numeric_cols[0]

                X = VE[[default_x]]
                y = VE[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            
                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)

                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

            # Private room
            MxP= dMx[(dMx["room_type"] == 2)]
            MP= dM[(dM["room_type"] == 2)]
            QP= dQ[(dQ["room_type"] == 2)]
            VP= dV[(dV["room_type"] == 2)]

            st.title("Regresión Lineal Private Room")
            col1, col2=  st.columns(2)

            with col1:
                st.subheader("México")
                default_x = "host_identity_verified" if "host_response_rate" in numeric_cols2 else numeric_cols[0]
                default_y = "host_response_rate" if "host_identity_verified" in numeric_cols2 else numeric_cols[0]

                X = MxP[[default_x]]
                y = MxP[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

                st.subheader("Malta")
                default_x = "bathrooms" if "accommodates" in numeric_cols2 else numeric_cols[0]
                default_y = "accommodates" if "bathrooms" in numeric_cols2 else numeric_cols[0]

                X = MP[[default_x]]
                y = MP[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")
                
            with col2:
                st.subheader("Quebec")
                default_x = "accommodates" if "price" in numeric_cols2 else numeric_cols[0]
                default_y = "price" if "accommodates" in numeric_cols2 else numeric_cols[0]

                X = QP[[default_x]]
                y = QP[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

                st.subheader("Victoria")
                default_x = "price" if "accommodates" in numeric_cols2 else numeric_cols[0]
                default_y = "accommodates" if "price" in numeric_cols2 else numeric_cols[0]

                X = VP[[default_x]]
                y = VP[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

            #Shared room
            MxS= dMx[(dMx["room_type"] == 3)]
            MS= dM[(dM["room_type"] == 3)]
            QS= dQ[(dQ["room_type"] == 3)]
            VS= dV[(dV["room_type"] == 4)]

            st.title("Regresión Lineal Shared Room")
            col1, col2=  st.columns(2)

            with col1:
                st.subheader("México")
                default_x = "host_response_rate" if "instant_bookable" in numeric_cols2 else numeric_cols[0]
                default_y = "instant_bookable" if "host_response_rate" in numeric_cols2 else numeric_cols[0]

                X = MxS[[default_x]]
                y = MxS[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

                st.subheader("Malta")
                default_x = "bathrooms" if "accommodates" in numeric_cols2 else numeric_cols[0]
                default_y = "accommodates" if "bathrooms" in numeric_cols2 else numeric_cols[0]

                X = MS[[default_x]]
                y = MS[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")
            
            with col2:
                st.subheader("Quebec")
                default_x = "host_response_rate" if "host_is_superhost" in numeric_cols2 else numeric_cols[0]
                default_y = "host_is_superhost" if "host_response_rate" in numeric_cols2 else numeric_cols[0]

                X = QS[[default_x]]
                y = QS[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

                st.subheader("Victoria")
                default_x = "bathrooms" if "accommodates" in numeric_cols2 else numeric_cols[0]
                default_y = "accommodates" if "bathrooms" in numeric_cols2 else numeric_cols[0]

                X = VS[[default_x]]
                y = VS[default_y]

                if len(X) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    fig, ax = plt.subplots()
                    ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                    ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                    ax.set_xlabel(default_x)
                    ax.set_ylabel(default_y)
                    ax.legend()
                    st.pyplot(fig)
                    
                    with st.container():
                        st.subheader("Resultados del Modelo")
                        col10, col11, col12= st.columns(3)
                        with col10:
                            st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                        with col11:
                            r2 = r2_score(y_test, y_pred)
                            st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                        with col12:
                            corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                            st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")
                else:
                    st.warning(f"No hay suficientes datos válidos para las variables en Victoria.")

            #Hotel room
            MxH= dMx[(dMx["room_type"] == 4)]
            MH= dM[(dM["room_type"] == 4)]
            QH= dQ[(dQ["room_type"] == 4)]
            VH= dV[(dV["room_type"] == 3)]

            st.title("Regresión Lineal Hotel Room")
            col1, col2=  st.columns(2)

            with col1:
                st.subheader("México")
                default_x = "bathrooms" if "accommodates" in numeric_cols2 else numeric_cols[0]
                default_y = "accommodates" if "bathrooms" in numeric_cols2 else numeric_cols[0]

                X = MxH[[default_x]]
                y = MxH[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

                st.subheader("Malta")
                default_x = "bathrooms" if "accommodates" in numeric_cols2 else numeric_cols[0]
                default_y = "accommodates" if "bathrooms" in numeric_cols2 else numeric_cols[0]

                X = MH[[default_x]]
                y = MH[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")
            
            with col2:
                st.subheader("Quebec")
                default_x = "host_response_rate" if "price" in numeric_cols2 else numeric_cols[0]
                default_y = "price" if "host_response_rate" in numeric_cols2 else numeric_cols[0]

                X = QH[[default_x]]
                y = QH[default_y]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(default_x)
                ax.set_ylabel(default_y)
                ax.legend()
                st.pyplot(fig)
                
                with st.container():
                    st.subheader("Resultados del Modelo")
                    col10, col11, col12= st.columns(3)
                    with col10:
                        st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                    with col11:
                        r2 = r2_score(y_test, y_pred)
                        st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                    with col12:
                        corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                        st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")

                st.subheader("Victoria")
                default_x = "host_response_rate" if "price" in numeric_cols2 else numeric_cols[0]
                default_y = "price" if "host_response_rate" in numeric_cols2 else numeric_cols[0]

                X = VH[[default_x]]
                y = VH[default_y]

                if len(X) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    fig, ax = plt.subplots()
                    ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                    ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                    ax.set_xlabel(default_x)
                    ax.set_ylabel(default_y)
                    ax.legend()
                    st.pyplot(fig)
                    
                    with st.container():
                        st.subheader("Resultados del Modelo")
                        col10, col11, col12= st.columns(3)
                        with col10:
                            st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                        with col11:
                            r2 = r2_score(y_test, y_pred)
                            st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                        with col12:
                            corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                            st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")
                else:
                    st.warning(f"No hay suficientes datos válidos para la variable '{default_x}' en Victoria.")
                


if pagina == "Mapa": 
    st.title ("Mapa de ubicaciones")
    ciudad = option_menu(
        menu_title=None,
        options=["CDMX", "Malta", "Quebec", "Victoria"],
        icons=["building", "sun", "cloud-snow", "tree"],
        orientation="horizontal", 
        styles={
            "icon": {"font-size": "24px"},
            "nav-link": {"--hover-color": "#D6EAF8"},
            "nav-link-selected": {"background-color": "#AED6F1"}
        }
    )
    
    if ciudad == "CDMX": 
         #Carga de dataframe original
        df_original = pd.read_csv("./dataset/Mexico.csv")
        df_original = df_original[
            (df_original["latitude"].between(19.2, 19.6)) &
            (df_original["longitude"].between(-99.3, -98.9))
        ]
        df=pd.read_csv("./dataset/MapMx.csv")
        df_original = df_original.dropna(subset=["latitude", "longitude"])

        #merge para recuperar coordenadas
        df_mapa = pd.merge(df[["id", "price", "accommodates"]], df_original[["id", "latitude", "longitude"]], on="id", how="inner")


        df_mapa = df_mapa.rename(columns={"latitude": "lat", "longitude": "lon"})

        # Filtramos ubicaciones válidas
        df_mapa = df_mapa.dropna(subset=["lat", "lon"])
        df_mapa = df_mapa[(df_mapa["lat"].between(19.2, 19.6)) & (df_mapa["lon"].between(-99.3, -98.9))]

        # Creación del mapa
        layer = pdk.Layer(
            "ScatterplotLayer",
            data = df_mapa, 
            get_position = '[lon, lat]',
            get_radius=100,
            pickable = True,
            get_fill_color="""
                [ 
                    price < 100 ? 0 : price < 200 ? 255 : 200,
                    price < 100 ? 200 : price < 200 ? 140 : 0,
                    150,
                    160
                ]
            """
        )

        #Configuramos la vista del mapa
        view_state = pdk.ViewState(
            latitude = df_mapa["lat"].mean(),
            longitude = df_mapa["lon"].mean(),
            zoom=10,
            pitch=0,
        )

        #creamos tooltip
        tooltip = {
            "html": "<b>Precio:</b> ${price} <br/><b>Acomoda:</b> {accommodates} personas",
            "style": {"backgroundColor": "white", "color": "black"}
        }

        # Mostramos el mapa
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v9"))
    
    if ciudad == "Malta": 
         #Carga de dataframe original
        df_original = pd.read_csv("./dataset/Malta.csv")
        df_original = df_original[
            (df_original["latitude"].between(35.8, 36.1)) &
            (df_original["longitude"].between(14.2, 14.6))
        ]

        df=pd.read_csv("./dataset/MapM.csv")
        df_original = df_original.dropna(subset=["latitude", "longitude"])

        #merge para recuperar coordenadas
        df_mapa = pd.merge(df[["id", "price", "accommodates"]], df_original[["id", "latitude", "longitude"]], on="id", how="inner")


        df_mapa = df_mapa.rename(columns={"latitude": "lat", "longitude": "lon"})

        # Filtramos ubicaciones válidas
        df_mapa = df_mapa.dropna(subset=["lat", "lon"])
        df_mapa = df_mapa[(df_mapa["lat"].between(35.8, 36.1)) & (df_mapa["lon"].between(14.2, 14.6))]


        # Creación del mapa
        layer = pdk.Layer(
            "ScatterplotLayer",
            data = df_mapa, 
            get_position = '[lon, lat]',
            get_radius=100,
            pickable = True,
            get_fill_color="""
                [ 
                    price < 100 ? 0 : price < 200 ? 255 : 200,
                    price < 100 ? 200 : price < 200 ? 140 : 0,
                    150,
                    160
                ]
            """
        )

        #Configuramos la vista del mapa
        view_state = pdk.ViewState(
            latitude = df_mapa["lat"].mean(),
            longitude = df_mapa["lon"].mean(),
            zoom=10,
            pitch=0,
        )

        #creamos tooltip
        tooltip = {
            "html": "<b>Precio:</b> ${price} <br/><b>Acomoda:</b> {accommodates} personas",
            "style": {"backgroundColor": "white", "color": "black"}
        }

        # Mostramos el mapa
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v9"))

    if ciudad == "Quebec": 
         #Carga de dataframe original
        df_original = pd.read_csv("./dataset/Quebec.csv")
        df_original = df_original[
            (df_original["latitude"].between(46.7, 47.0)) &
            (df_original["longitude"].between(-71.4, -71.0))
        ]

        df_original = df_original.dropna(subset=["latitude", "longitude"])
        df=pd.read_csv("./dataset/MapQ.csv")
        #merge para recuperar coordenadas
        df_mapa = pd.merge(df[["id", "price", "accommodates"]], df_original[["id", "latitude", "longitude"]], on="id", how="inner")

        df_mapa = df_mapa.rename(columns={"latitude": "lat", "longitude": "lon"})

        # Filtramos ubicaciones válidas
        df_mapa = df_mapa.dropna(subset=["lat", "lon"])
        df_mapa = df_mapa[(df_mapa["lat"].between(46.7, 47.0)) & (df_mapa["lon"].between(-71.4, -71.0))]

        # Creación del mapa
        layer = pdk.Layer(
            "ScatterplotLayer",
            data = df_mapa, 
            get_position = '[lon, lat]',
            get_radius=100,
            pickable = True,
            get_fill_color="""
                [ 
                    price < 100 ? 0 : price < 200 ? 255 : 200,
                    price < 100 ? 200 : price < 200 ? 140 : 0,
                    150,
                    160
                ]
            """
        )

        #Configuramos la vista del mapa
        view_state = pdk.ViewState(
            latitude = df_mapa["lat"].mean(),
            longitude = df_mapa["lon"].mean(),
            zoom=10,
            pitch=0,
        )

        #creamos tooltip
        tooltip = {
            "html": "<b>Precio:</b> ${price} <br/><b>Acomoda:</b> {accommodates} personas",
            "style": {"backgroundColor": "white", "color": "black"}
        }

        # Mostramos el mapa
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v9"))

    if ciudad == "Victoria": 
         #Carga de dataframe original
        df_original = pd.read_csv("./dataset/Victoria.csv")
        df_original = df_original[
            (df_original["latitude"].between(48.3, 48.6)) &
            (df_original["longitude"].between(-123.6, -123.2))
        ]
        df=pd.read_csv("./dataset/MapV.csv")
        df_original = df_original.dropna(subset=["latitude", "longitude"])

        #merge para recuperar coordenadas
        df_mapa = pd.merge(df[["id", "price", "accommodates"]], df_original[["id", "latitude", "longitude"]], on="id", how="inner")


        df_mapa = df_mapa.rename(columns={"latitude": "lat", "longitude": "lon"})

        # Filtramos ubicaciones válidas
        df_mapa = df_mapa.dropna(subset=["lat", "lon"])
        df_mapa = df_mapa[(df_mapa["lat"].between(47, 49)) & (df_mapa["lon"].between(-125, -122))]

        # Creación del mapa
        layer = pdk.Layer(
            "ScatterplotLayer",
            data = df_mapa, 
            get_position = '[lon, lat]',
            get_radius=100,
            pickable = True,
            get_fill_color="""
                [ 
                    price < 100 ? 0 : price < 200 ? 255 : 200,
                    price < 100 ? 200 : price < 200 ? 140 : 0,
                    150,
                    160
                ]
            """
        )

        #Configuramos la vista del mapa
        view_state = pdk.ViewState(
            latitude = df_mapa["lat"].mean(),
            longitude = df_mapa["lon"].mean(),
            zoom=10,
            pitch=0,
        )

        #creamos tooltip
        tooltip = {
            "html": "<b>Precio:</b> ${price} <br/><b>Acomoda:</b> {accommodates} personas",
            "style": {"backgroundColor": "white", "color": "black"}
        }

        # Mostramos el mapa
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v9"))
    
    
