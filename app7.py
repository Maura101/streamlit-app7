import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
import plotly.express as px
import pynarrative as pn

st.set_page_config(page_title="Cancerogenicità e relazioni con le caratteristiche molecolari", layout="wide")

st.title("🧬 Cancerogenicità e relazioni con le caratteristiche molecolari 🧬")
st.markdown("""
Studente: Maura Ruggiero
    """)

st.image("https://www.thoughtco.com/thmb/fnRlrPTo08rb9Cdfn0gMg7VSbY4=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/molecular-model-illustration-545862141-5782ca213df78c1e1f55b395.jpg")


st.markdown("""
Ogni giorno, siamo esposti a centinaia di sostanze chimiche: nei cibi che mangiamo, nei farmaci che assumiamo, nei prodotti che usiamo. Ma quali di queste molecole sono potenzialmente cancerogene? E soprattutto… **possiamo prevederlo a partire dalle loro proprietà chimico-fisiche?** 🔬

In questo progetto esploriamo un dataset (3678 osservazioni, 57 variabili, inizialmente) di molecole tendenzialmente tossiche classificate per **cancerogenicità**, mettendo in relazione la loro **struttura**, **massa molecolare**, **polarità**, **stato fisico**, **origine** (naturale o sintetica) e altre caratteristiche computazionali. L’obiettivo è **identificare pattern nascosti**, capire **quali proprietà molecolari sono associate a un rischio più elevato**, e gettare le basi per **modelli predittivi più robusti**.

📊 Attraverso grafici interattivi, correlazioni e analisi esplorative, cerchiamo risposte a domande come:

* Esistono pattern strutturali comuni alle molecole cancerogene?
* Le molecole più grandi sono davvero più cancerogene?
* C’è un legame tra polarizzabilità, rifrazione e cancerogenicità?
* Quanto conta l’origine della molecola?

🔍 Questo progetto unisce il rigore della **tossicologia computazionale** con la potenza dell’**analisi dei dati**, per offrire una visione generale e intuitiva dei fattori che rendono una molecola più o meno cancerogena.

    """)

# --- Caricamento dati ---
@st.cache_data
def load_data():
    return pd.read_csv("df_clean.csv")

df_clean = load_data()

# --- Visualizzo dati ---
st.subheader("Visualizza i dati")
if st.checkbox("Mostra prime righe del dataset"):
    st.dataframe(df_clean.head())

# Mappo i valori numerici in etichette descrittive
df_clean['carcinogenicity_label'] = df_clean['carcinogenicity_score'].map({
    0: '0 = non classificato come cancerogeno',
    1: '1 = cancerogeno o possibile cancerogeno'
})

# --- Istogramma con Plotly e Streamlit ---
st.subheader("Distribuzione delle classi di cancerogenicità")
fig_plotly = px.histogram(
    df_clean,
    x='carcinogenicity_label',
    color='carcinogenicity_label',
    color_discrete_sequence=['#66c2a5', '#fc8d62'],
    title="Distribuzione delle classi di cancerogenicità"
)
fig_plotly.update_layout(
    xaxis_title='Classe di cancerogenicità',
    yaxis_title='Conteggio',
    legend_title='Classe'
)
st.plotly_chart(fig_plotly)



# --- Correlazione tra variabili numeriche ---
with st.expander("📉 Matrice di correlazione tra variabili numeriche"):
    colss = ['moldb_average_mass', 'JCHEM_ACCEPTOR_COUNT',
        'JCHEM_AVERAGE_POLARIZABILITY', 'JCHEM_BIOAVAILABILITY',
        'JCHEM_DONOR_COUNT', 'JCHEM_FORMAL_CHARGE', 'JCHEM_GHOSE_FILTER',
        'JCHEM_LOGP', 'JCHEM_MDDR_LIKE_RULE', 'JCHEM_NUMBER_OF_RINGS',
        'JCHEM_PHYSIOLOGICAL_CHARGE', 'JCHEM_POLAR_SURFACE_AREA',
        'JCHEM_REFRACTIVITY', 'JCHEM_ROTATABLE_BOND_COUNT',
        'JCHEM_RULE_OF_FIVE', 'JCHEM_VEBER_RULE', 'carcinogenicity_score']
    
    cm = np.corrcoef(df_clean[colss].dropna().values.T)
    fig_corr = plt.figure(figsize=(16, 16))
    heatmap(cm, row_names=colss, column_names=colss, cell_font_size=7)
    fig_corr = plt.gcf()
    st.pyplot(fig_corr)


# --- Pynarrative ---
# Calcola le statistiche
stats = df_clean.describe()

# Mostrale in Streamlit
st.subheader("📊 Statistiche descrittive delle variabili numeriche")
st.dataframe(stats)

mean_mass = df_clean["moldb_average_mass"].mean()
mean_logp = df_clean["JCHEM_LOGP"].mean()
mean_refractivity = df_clean["JCHEM_REFRACTIVITY"].mean()
perc_exogenous = df_clean[df_clean["origin"] == "Exogenous"].shape[0] / df_clean.shape[0] * 100
perc_carc = df_clean[df_clean["carcinogenicity_score"] == 1].shape[0] / df_clean.shape[0] * 100

story_text = f"""
L'analisi del dataset ha rivelato alcune caratteristiche chiave delle molecole in relazione alla loro potenziale cancerogenicità.

📊 **Complessivamente**, il valore medio della massa molecolare è di circa {mean_mass:.1f} g/mol, indicando la presenza di molecole di dimensioni medio-alte nel dataset. Il valore medio di LogP è pari a {mean_logp:.2f}, suggerendo un grado di lipofilia moderato: molte molecole hanno quindi la potenzialità di attraversare facilmente le membrane cellulari. La rifrazione molare media, un indice della polarizzabilità elettronica, è {mean_refractivity:.2f}, coerente con la presenza di molecole complesse.

🧪 **Dal punto di vista tossicologico**, il dataset mostra che circa il {perc_carc:.1f}% delle molecole sono classificate come cancerogene o potenzialmente tali. Inoltre, il {perc_exogenous:.1f}% delle molecole ha origine esogena, sottolineando come molte di esse possano derivare da sostanze industriali, contaminanti ambientali o farmaci.

📈 Un'osservazione interessante riguarda le relazioni tra le variabili: molecole con massa molecolare elevata tendono ad avere anche valori alti di rifrazione, il che è chimicamente plausibile poiché una maggiore massa comporta una struttura più complessa e quindi più facilmente polarizzabile.

🧬 Le molecole cancerogene tendono a concentrarsi nella zona del grafico che mostra alti valori sia di massa che di rifrazione molare. Questo suggerisce che molecole più pesanti e polarizzabili possano avere una maggiore capacità di interazione con bersagli biologici, come DNA o proteine cellulari, aumentando il rischio di effetti mutageni.

💡 Tuttavia, è importante sottolineare che **nessuna singola variabile** è risultata fortemente predittiva della cancerogenicità. Piuttosto, il rischio sembra emergere da un insieme di caratteristiche strutturali, confermando la necessità di un approccio multivariato per una valutazione più accurata.

L'esplorazione dei dati mostra, dunque, che la cancerogenicità non dipende da un solo fattore, ma è il risultato di **interazioni sinergiche tra massa, polarità, stato fisico, lipofilia e origine molecolare**.
"""
story = pn.Story(story_text)
with st.expander("🧠 Narrazione con Pynarrative"):
    st.markdown(story_text, unsafe_allow_html=True)


# --- Scatterplot ---
with st.expander("🧬 Relazioni tra variabili molecolari e cancerogenicità"):

    fig_scatter = px.scatter(df_clean, x='moldb_average_mass', y='carcinogenicity_score',
                             color='origin',
                             size='JCHEM_ACCEPTOR_COUNT',
                             hover_data=['JCHEM_LOGP', 'JCHEM_DONOR_COUNT'],
                             title='Relazione tra peso molecolare, origine e cancerogenicità')
    st.plotly_chart(fig_scatter)

    
    fig_scatter2 = px.scatter(df_clean, x='moldb_average_mass', y='carcinogenicity_score',
                 color='state',  
                 size='JCHEM_POLAR_SURFACE_AREA',  
                 hover_data=['JCHEM_LOGP', 'JCHEM_POLAR_SURFACE_AREA'],
                 title='Relazione tra peso molecolare, stato fisico della molecola e cancerogenicità')
    st.plotly_chart(fig_scatter2)

# --- Boxplot e Istogrammi ---
with st.expander("📉 Boxplot e Istogrammi per variabili molecolari"):
    sns.set(style="whitegrid", palette="muted", font_scale=1.1)
    vars_molecolari = ['moldb_average_mass', 'JCHEM_LOGP']

    for col in vars_molecolari:
        fig_h, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df_clean, x=col, kde=True, hue='carcinogenicity_score',
                     multiple="stack", palette='deep', ax=ax)
        ax.set_title(f'Distribuzione di {col} per classe di cancerogenicità')
        st.pyplot(fig_h)

        fig_b, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df_clean, x='carcinogenicity_score', y=col, palette='Set2', ax=ax)
        ax.set_title(f'{col} per classe di cancerogenicità (Boxplot)')
        st.pyplot(fig_b)


# --- Boxplot ---
with st.expander("🧬 Scegli una caratteristica molecolare"):
    fig_box = px.scatter(df_clean, x='moldb_average_mass', y='JCHEM_REFRACTIVITY',
                 color='carcinogenicity_score',
                 hover_name='common_name',
                 hover_data=['state', 'origin', 'JCHEM_LOGP'])

    st.plotly_chart(fig_box)



with st.expander("🧠 Interpretazione dei risultati e conclusioni"):
    st.markdown("""
    **Interpretazione della matrice di correlazione delle variabili numeriche**

La matrice di correlazione mostra che nessuna singola variabile numerica è fortemente correlata con la variabile target. Le correlazioni più alte sono deboli e positive: L'area superficiale polare (+0.20), la massa media molecolare (+0.11) e la rifrazione molare mostrano una tendenza secondo cui molecole più grandi e polari potrebbero avere una maggiore probabilità di essere classificate come cancerogene.

Tuttavia, queste correlazioni sono **troppo basse per essere considerate predittori affidabili singolarmente**, suggerendo che la cancerogenicità dipende da un insieme più complesso di caratteristiche.

Sono invece presenti **forti correlazioni positive tra alcune variabili indipendenti**, che spiegano come una massa molecolare più importante, sia spesso associata ad una rifrazione molare e ad una polarizzabilità molecolare più alte e che inoltre sia correlata positivamente con il numero di anelli che compongono la molecola. Infatti spesso le molecole con maggiore massa hanno più elettroni e più legami, quindi sono più polarizzabili, e questo aumenta la rifrazione molare; molecole più grandi tendenzialmente hanno inoltre più strutture cicliche (anelli), come quelli aromatici o eterociclici e rispondono più facilmente ai campi elettrici esterni.

In conclusione, l’analisi suggerisce la presenza di correlazioni positive alte tra alcune variabili riguardanti la struttura chimico-fisica delle molecole ed è inoltre evidente come la cancerogenicità non possa essere spiegata da singole proprietà molecolari ma richieda un approccio multivariato.
    """)

    
    st.markdown("""
    **Interpretazione del grafico: Relazione tra peso molecolare, origine e carcinogenicità**

    - Le sostanze esogene mostrano una chiara tendenza ad avere un punteggio cancerogeno elevato (c'era da aspettarselo! 🧠).
    - Il peso molecolare elevato, come vedremo anche in seguito, potrebbe essere un fattore di rischio delle capacità cancerogene di una molecola.
    """)

    

    st.markdown("""
    **Interpretazione del grafico: Relazione tra peso molecolare, stato fisico e carcinogenicità**

    Questo grafico mostra la relazione tra **cancerogenicità** (asse y) e **peso molecolare** (asse x), con il colore che rappresenta lo **stato fisico** in cui si presenta la molecola.

    - Le molecole **solide** (azzurro) e **liquide** (blu) sono le più frequenti.
    - Le **molecole solide** con **peso molecolare elevato (>1000)** tendono ad avere un `carcinogenicity_score = 1`, quindi **sono cancerogene o potenzialmente tali**.
    - Le **molecole liquide** sono distribuite più uniformemente tra cancerogene e non.
    - Le classi meno rappresentate come **Gas**, **Solid or Liquid**, **Gas or Liquid** sono troppo poche per trarre conclusioni affidabili.
    - Il fatto che i punti si trovino solo a `0` o `1` sull’asse y conferma che il punteggio è binario.

    **Conclusione:** lo **stato fisico da solo** sembra **non essere un forte predittore** di cancerogenicità, ma **quando associato a un peso molecolare elevato** (es. molecole solide pesanti), può indicare **maggiore rischio**.
    """)


    st.markdown("""
    **Interpretazione del grafico boxplot: Scegli una variabile**

📈 Pattern osservati
È visibile una chiara correlazione positiva tra massa molecolare e rifrazione molare: le molecole più pesanti tendono ad avere anche una rifrazione più alta.
Questo è chimicamente plausibile, poiché molecole più grandi hanno più elettroni e strutture più complesse, il che le rende più polarizzabili e quindi con maggiore rifrazione molare.

Questa relazione è confermata dalla distribuzione diagonale crescente visibile nel grafico.

🧪 Legame con la cancerogenicità
I punti più scuri (carcinogenicity_score = 1) si trovano prevalentemente nella parte superiore destra del grafico: quindi molecole con massa e rifrazione elevate sono più spesso cancerogene.

Questo suggerisce che strutture molecolari complesse, tipiche di molecole con elevata rifrazione e massa, possono avere una maggiore capacità di interazione con bersagli biologici (es. DNA, proteine cellulari) e inoltre che possono essere anche più lipofile, attraversare più facilmente le membrane biologiche e accumularsi nei tessuti.

⚠️ Considerazioni aggiuntive
Il fatto che le molecole con carcinogenicity_score = 1 non si trovino solo in un punto specifico, ma abbiano una certa dispersione, conferma che massa e rifrazione da sole non sono predittori univoci, ma concorrono con altre variabili (es. LogP, numero di anelli, polarità) alla determinazione della cancerogenicità.
Alcuni outlier cancerogeni appaiono anche in aree con massa relativamente bassa, suggerendo l’esistenza di meccanismi alternativi di tossicità anche per molecole più piccole. Gli outlier con massa alta,rifrazione alta e classificati come cancerogeni o possibilmente cancerogeni risultano essere
    """)

    

    st.markdown("""
 ✅ **Considerazioni e conclusioni finali**
 
L’analisi esplorativa condotta su questo dataset ci ha permesso di guardare dentro la chimica delle molecole con una lente statistica. E ciò che emerge è chiaro: **non esiste un singolo "colpevole" della cancerogenicità**, ma piuttosto una rete di fattori intrecciati che insieme concorrono a determinarla.

💡 Iniziamo da un dato cruciale: il dataset è **sbilanciato**, con circa **il doppio delle molecole non cancerogene** rispetto a quelle cancerogene o potenzialmente tali. Questo riflette la realtà biologica ma impone attenzione nell’interpretazione dei dati.

📊 La **matrice di correlazione** conferma che **nessuna variabile numerica** – da sola – spiega il fenomeno cancerogeno. Le correlazioni più alte, seppur deboli, sono positive: molecole con **massa molecolare maggiore**, **superficie polare più estesa** e **rifrazione molare più elevata** tendono ad avere un punteggio cancerogeno più alto. Questo ci suggerisce che molecole **più grandi, più polari e strutturalmente complesse** potrebbero avere maggiori probabilità di interagire con strutture biologiche sensibili, come il DNA.

🔬 Dal punto di vista chimico-fisico, queste variabili sono intercorrelate: **più una molecola è grande**, **più tende ad avere legami multipli**, **più è polarizzabile**, e **più aumenta la sua rifrazione molare**. Inoltre, **molecole pesanti** tendono ad avere **più anelli aromatici o eterociclici**, strutture spesso associate a potenziale mutageno. In sintesi, la **complessità strutturale** può tradursi in **maggiore reattività biologica**.

🌍 Anche la **provenienza della molecola** conta: le **sostanze esogene**, spesso derivate da sintesi industriale o inquinanti ambientali, mostrano una maggiore incidenza di classificazione cancerogena. Non è un caso: molte di queste molecole sono **xenobiotici**, cioè sostanze estranee all'organismo, che spesso **sfuggono ai meccanismi di detossificazione** o **generano metaboliti reattivi**.

🧊 E per quanto riguarda lo **stato fisico**? Da solo non basta a spiegare la cancerogenicità, ma in **combinazione con la massa molecolare** può diventare rilevante. Le **molecole solide e pesanti** sembrano avere un rischio maggiore. Questo potrebbe essere legato alla loro **persistenza nell'ambiente**, alla **difficoltà di degradazione** e alla tendenza ad **accumularsi nei tessuti biologici**.

🔗 Le **forti correlazioni tra variabili strutturali**, come massa, rifrazione, polarizzabilità e numero di anelli, ci raccontano un’altra verità: **la tossicità è multiforme**. Non può essere spiegata linearmente. Serve un approccio **multivariato, integrato, sistemico**.

🧬 In conclusione, questa analisi suggerisce che:

* La **cancerogenicità molecolare** è il risultato di **interazioni sinergiche** tra molteplici caratteristiche: **dimensione**, **polarità**, **forma**, **origine**, **persistenza** in alcuni casi correlate proporzionalmente tra di loro.
* Non esiste un unico predittore “magico”, ma **un profilo molecolare complesso** da interpretare nel suo insieme.
* Questo riflette ciò che già la tossicologia molecolare ci insegna: la pericolosità di una sostanza **nasce dalla sua struttura**, **dalle sue proprietà fisico-chimiche**, e **dalla sua capacità di interagire selettivamente con bersagli biologici**.

📈 Da qui, il passo successivo è naturale: per valutare in modo affidabile il rischio cancerogeno serve un approccio **modellistico e predittivo**, basato su **dati multidimensionali** e metodi avanzati.

🔍 L’esplorazione dei dati condotta in questo lavoro è solo il punto di partenza. Ma è già un passo fondamentale verso uno **studio della cancerogenicità e delle sostanze nocive più trasparente, interpretabile e guidata dai dati**.

    """)
