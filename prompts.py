PROMPTS = {
    "de": {
        "summary_system_prompt": \
            """
                Du bist ein Data Science-Experte und analysierst einen tabellarische Datensatz zusammen mit seinen Metadaten.
                Alle Aussagen sollen sich strikt auf den Datensatz beziehen. Wichtig sind Auffälligkeiten, Trends, fehlende Werte, Datentypen, Zusammenhänge. 
                Antwortformat soll wissenschaftlich sein. Sprich für Laien verständlich, aber dennoch informativ und verhalte dich als Experte.
                Auch ein Experte soll aus den bereitgestellten Daten und Informationen einen Nutzen haben.
                Sei dir bei sämtlichen Aussagen sicher und beziehe dich immer auf den Datensatz oder andere durch den Prompt bereitgestellte Informationen.

                Folgende Daten sollen bei der Analyse helfen:
                - Columns: '{column_names}'
                - Descriptions: '{descriptions}'
                - Metadata: '{metadata}'
            """,
        "summary_user_prompt": \
            """
                Fasse mir (Experte) den Datensatz passend zusammen.
                Ich möchte als Ergebnis eine Zusammenfassung bzw. Erklärung des Datensatzes und eine Erklärung / Beschreibung der einzelnen Spalten des Datensatzes.
                Die verwendete Sprache soll Deutsch sein.
                Für mich sind vor allem Visualisierungen wichtig, die mir helfen den Datensatz zu verstehen und zu analysieren. 
            """,
        "generate_python_code": \
            """
                Erzeuge mir basierend auf der vorherigen Zusammenfassung und der Datenstruktur Python-Code,
                der eine explorative Datenanalyse (EDA) des Datensatzes durchführt und passende Visualisierungen erstellt.

                Die Daten können mit folgendem Befehl geladen werden:
                `df = pd.read_csv("'{dataset_path}'", sep="'{dataset_sep}'")`

                Vorgaben für den Code:
                - Verwende ausschließlich `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `geopandas`, `basemap`.
                - Der Code soll modular, gut kommentiert und direkt ausführbar sein, ohne syntaktische Fehler.
                - Alle Diagramme sollen optisch ansprechend, gut beschriftet (in Deutsch), lesbar und in PNG-Dateien gespeichert werden unter:
                  `./output/<plot_name>.png`
                - Wähle Diagrammtypen entsprechend der Datenbedeutung:
                  - Geographische Variablen → räumliche Verteilung (z.B. Karte mit Markierung der Punkte).
                  - Zeitliche Variablen → Untersuchen ob sich ein zeitlicher Verlauf einer anderen Variable abbilden lässt.
                  - Numerische Variablen → Histogramme, Boxplots und Scatterplots für Zusammenhänge.
                  - Kategorische Variablen → Balkendiagramme der Häufigkeitsverteilung (ggf. Top 10 für lange Listen).
                - Führe auch kurze statistische Analysen durch, gegebenen falls mit Visualisierung:
                  - Anteil fehlender Werte je Spalte,
                  - Korrelationen numerischer Variablen,
                  - Übersichtstabellen zu zentralen Kennwerten (Mittelwert, Standardabweichung etc.).
                - Verwende Farben, Beschriftungen und Titel sinnvoll:
                  - Titel sollen beschreiben, was gezeigt wird (auf Deutsch),
                  - Legenden und Achsenbeschriftungen sollen keine Information abschneiden,
                  - Achsen in SI-Einheiten oder sinnvollen Skalen beschriften.
                - Füge kurze erklärende Kommentare hinzu, **warum** bestimmte Visualisierungen sinnvoll sind.
                - Priorisiere Plot-Typen, die einem Data-Science-Workflow entsprechen (Datenqualität, Verteilung, Beziehung, Geografie, Zeit).
                - Es soll bei allen Berechnungen und Plots beachtet und berücksichtigt werden, dass fehlende Werte und auch String und Boolean Werte im Datensatz vorhanden sind. Also entsprechend damit umgehen.
                - Der zurückgegebene Code soll bitte in UTF-8 kodiert sein.
                - Für jede Visualisierung soll eine separate Methode erstellt werden, welche am Ende des Skripte mit try except ausgeführt wird. Die Fehler Meldung soll ausgegeben werden, aber die Ausführung des restlichen Codes soll nicht abgebrochen werden.

                Zum besseren Verständnis:
                Das ist das Ergebnis von `df.head(10)`:
                '{df_head_markdown}'
            """,
        "generate_r_code": \
            """
                Erzeuge mir basierend ein R-Skript, das eine explorative Datenanalyse (EDA) des Datensatzes durchführt und passende Visualisierungen erstellt.

                Die Daten können aus folgender CSV geladen werden:
                - Pfad zur CSV Datei: `'{dataset_path}'`
                - Trennzeichen: `'{dataset_sep}'`

                Vorgaben für den Code:
                - Der Code soll modular, gut kommentiert und direkt ausführbar sein, ohne syntaktische Fehler.
                - Alle Diagramme sollen optisch ansprechend, gut beschriftet (in Deutsch), lesbar und in PNG-Dateien gespeichert werden unter:
                  `./output/<plot_name>.png`
                - Wähle Diagrammtypen entsprechend der Datenbedeutung. Dies sind Hinweise, überlege selber ob die Hinweise für die entsprechenden Spalten passend sind:
                  - Geographische Variablen → räumliche Verteilung (z.B. Karte mit Markierung der Punkte).
                  - Zeitliche Variablen → Untersuchen ob sich ein zeitlicher Verlauf einer anderen Variable abbilden lässt.
                  - Numerische Variablen → Histogramme, Boxplots und Scatterplots für Zusammenhänge.
                  - Kategorische Variablen → Balkendiagramme der Häufigkeitsverteilung (ggf. Top 10 für lange Listen).
                - Führe auch kurze statistische Analysen durch, falls sie sich anbieten und dsinnvoll sind. Gegebenen falls mit Visualisierung:
                  - Anteil fehlender Werte je Spalte,
                  - Korrelationen numerischer Variablen,
                  - Übersichtstabellen zu zentralen Kennwerten (Mittelwert, Standardabweichung etc.).
                - Verwende Farben, Beschriftungen und Titel sinnvoll:
                  - Titel sollen beschreiben, was gezeigt wird (auf Deutsch),
                  - Legenden und Achsenbeschriftungen sollen keine Information abschneiden,
                  - Achsen in Einheiten oder sinnvollen Skalen beschriften.
                - Es soll bei allen Berechnungen und Plots beachtet und berücksichtigt werden, dass fehlende Werte und auch String und Boolean Werte im Datensatz vorhanden sind. Also entsprechend damit umgehen.
                - Der zurückgegebene Code soll bitte in UTF-8 kodiert sein.
                - Für jede Visualisierung soll eine separate Methode erstellt werden, welche eine passende Fehlerbehandlung hat. Die Fehler Meldung soll ausgegeben werden, aber die Ausführung des restlichen Codes soll nicht abgebrochen werden.

                Zum besseren Verständnis:  
                Das ist das Ergebnis von `df.head(10)` auf den Datensatz:
                '{df_head_markdown}'

                Das ist das Ergebnis der vorherigen Analyse der einzelnen Spalten, beziehe diese Informationen in der Entscheidung mit ein, welche Diagramme sinnvoll sind:
                '{summary_columns}'

                Das ist eine Beschreibung des Datensatzes:
                '{summary}'
            """,
        "decide_regenerate_code_system_prompt": \
            """
                Du bist ein Experte darin Code Output zu interpretieren, der entscheidet, ob der gegebene Text Fehler enthält, die eine erneute Generierung des Codes erforderlich machen.
                Antworte mit einer bool Antwort, welche true ist, genau dann wenn der Code Fehler enthält, die eine erneute Erzeugung des Codes zwingend notwendig machen.
                Ansonsten antworte mit false.
                Wichtig, du bekommst den Text von stdout und stderr des Codes. Das heißt gegebenenfalls sind dort auch nur Infos oder Deprecated Warnings enthalten, diese musst du von wahren Fehlern bzw. Exceptions unterscheiden, welche unbedingt korrigiert werden müssen damit ein Diagramm erzeugt werden kann und den restlichen Ablauf des Skriptes nicht behindern.
            """,
        "decide_regenerate_code_user_prompt": \
            """
                Hier ist die Ausgabe (stdout) und die Fehlerausgabe (stderr) des Codes:
                stdout:
                '{test_stdout}'

                stderr:
                '{test_stderr}'

                Bitte entscheide, ob der Code unbedingt neu generiert werden muss.
            """,
        "regenerate_code_user_prompt":
            """
                Der vorherige Code hatte folgende Fehler:
                stdout:
                '{test_stdout}'
                stderr:
                '{test_stderr}'
                Bitte generiere den Code erneut und behebe die oben genannten Fehler.
                Das ist die Beschreibung des Codes:
                '{code_explanation}'
                Das ist der vorherige Code:
                '{code}'
            """
    },
    "en": {
        "summary_system_prompt": \
            """
                You are a data science expert helping me analyze a tabular dataset.
                Please provide a summary of the dataset: what it is about, any notable aspects, trends, or missing values. 
                The response should be easy to understand for general users,
                while still informative enough for an expert to judge whether the dataset is suitable for their work.

                Here is all relevant information:
                - Columns: '{column_names}'
                - Descriptions: '{descriptions}'
                - Metadata: '{metadata}'
            """,
        "summary_user_prompt": \
            """
                Please summarize the dataset appropriately.
                I want a summary or explanation of the dataset and a description of each column.
                The response language should be English.
            """,
        "generate_python_code": \
            """
                Based on the previous summary and the data structure, generate Python code
                that performs an exploratory data analysis (EDA) of the dataset and produces suitable visualizations.

                The data can be loaded with:
                `df = pd.read_csv("'{dataset_path}'", sep="'{dataset_sep}'")`

                Code requirements:
                - Use only `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `geopandas`, and `basemap`.
                - The code must be modular, well‑commented, directly executable, and free of syntax errors.
                - All plots must be visually appealing, clearly labeled (in English), readable, and saved as PNG files under:
                  `./output/<plot_name>.png`
                - Choose chart types based on data meaning:
                  - Geographic variables → spatial distribution (e.g., map with point markers)
                  - Temporal variables → investigate whether temporal trends relate to other variables
                  - Numerical variables → histograms, boxplots, and scatterplots
                  - Categorical variables → bar plots of frequency distributions (top-10 for long lists if needed)
                - Include brief statistical analyses when appropriate, possibly with visualizations:
                  - proportion of missing values per column
                  - correlations of numerical variables
                  - summary tables for key metrics (mean, std-dev, etc.)
                - Use colors, labels, and titles meaningfully:
                  - Titles should describe what is shown (in English)
                  - Legends and axis labels must not truncate information
                  - Axes should be labeled in SI units or meaningful scales
                - Add concise explanatory comments about **why** certain plots are used.
                - Prioritize visualizations consistent with a data‑science workflow (data quality, distribution, relationships, geography, time).
                - Make sure missing values, strings, and booleans in the dataset are handled appropriately.
                - The returned code must be UTF‑8 encoded.
                - Create a separate function for each visualization and execute them at the end of the script within try/except blocks so that errors are logged but do not stop execution of the rest of the code.

                For reference, this is the output of `df.head(10)`:
                '{df_head_markdown}'
            """,
        "generate_r_code": \
            """
                Based on the analysis, generate an R script that performs an exploratory data analysis (EDA)
                of the dataset and produces meaningful visualizations.

                The dataset can be loaded from the following CSV:
                - File path: `'{dataset_path}'`
                - Separator: `'{dataset_sep}'`

                Code requirements:
                - The code must be modular, well‑commented, directly runnable, and free of syntax errors.
                - All plots must be visually appealing, clearly labeled (in English), readable, and saved as PNG files under:
                  `./output/<plot_name>.png`
                - Choose chart types according to the data’s meaning. The following hints may help, but apply your judgment:
                  - Geographic variables → spatial distribution (e.g., map with points)
                  - Temporal variables → study if variable values change over time
                  - Numerical variables → histograms, boxplots, scatterplots
                  - Categorical variables → bar charts for frequency distributions (top-10 when appropriate)
                - Add light statistical summaries if they make sense, optionally with visualization:
                  - proportion of missing values per column
                  - correlations between numeric variables
                  - summary tables of central measures (mean, standard deviation, etc.)
                - Use colors, labels, and titles sensibly:
                  - Titles should clearly state what is shown (in English)
                  - Legends and axes must not cut off information
                  - Axes in meaningful units or scales
                - Handle missing values, string, and boolean types gracefully in all computations and plots.
                - The returned code must be UTF‑8 encoded.
                - Each visualization should have its own function with proper error handling.
                  Errors should be printed but must not interrupt the execution of the rest of the script.

                For reference:  
                This is the output of `df.head(10)` on the dataset:
                '{df_head_markdown}'

                This is the result of the previous column‑wise analysis — use this information to decide which visualizations are meaningful:
                '{summary_columns}'

                This is a general description of the dataset:
                '{summary}'
            """,
        "decide_regenerate_code_system_prompt": \
            """
                You are an expert in interpreting code output.
                Determine whether the given text contains errors that make regeneration of the code necessary.
                Answer with a boolean: true if the code contains critical errors that require regeneration,
                false otherwise.
                Note: You receive the content of stdout and stderr.
                Some lines may contain informational or deprecated warnings — these must be distinguished from real exceptions
                that prevent a plot from being created or block the rest of the script.
            """,
        "decide_regenerate_code_user_prompt": \
            """
                Here is the program output (stdout) and the error output (stderr):

                stdout:
                '{test_stdout}'

                stderr:
                '{test_stderr}'

                Please decide whether the code absolutely needs to be regenerated.
            """,
        "regenerate_code_user_prompt":
            """
                The previous code produced the following errors:
                stdout:
                '{test_stdout}'
                stderr:
                '{test_stderr}'

                Please regenerate the code and fix the errors above.
                This is the description of the code:
                '{code_explanation}'
                This is the previous code:
                '{code}'
            """
    }
}


def get_prompt(language: str, key: str, **fmt):
    return PROMPTS[language][key].format(**fmt)
