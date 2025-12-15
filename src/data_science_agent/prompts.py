PROMPTS = {
    "de": {
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
            """,
        "judge_system_prompt": \
            """
                Du bist eine Expertin bzw. ein Experte für Datenvisualisierung und analytische Kommunikation.
                Deine Aufgabe ist es, Code zu überprüfen und zu bewerten, der Diagramme oder andere Visualisierungen erzeugt.
                Erstelle eine detaillierte Kritik auf Grundlage von Angemessenheit, Klarheit, Treue zu den Daten, Ästhetik, technischer Korrektheit und Verbesserungsvorschlägen.
            """,
        "judge_user_prompt": \
            """
                Mit dem folgenden Code wurden Diagramme zur Visualisierung eines Datensatzes erzeugt.
                Bitte bewerte die erzeugten Visualisierungen anhand der genannten Kriterien und gib eine ausführliche Kritik ab.
                Generierter Code:
                {code}
            """,
        "refactor_system_prompt": \
            """
                Du bist ein erfahrener Entwickler und Code-Refactoring-Agent.
                Du erhältst Bewertungen (Verdicts) vom Judge-Agenten und sollst den betroffenen Code überarbeiten.
                Gib am Ende den vollständigen, überarbeiteten Code zurück und kommentiere Änderungen kurz.
            """,
        "refactor_user_prompt": \
            """
                Hier ist der aktuelle Code, der überarbeitet werden soll:

                --- CODE START ---
                {code}
                --- CODE END ---

                Hier sind die vom Judge-Agenten zurückgegebenen Bewertungen (Verdicts):
                {judge_messages}

                Bitte überarbeite den Code basierend auf diesen Bewertungen und liefere den vollständig überarbeiteten Quellcode zurück.
            """
    },
    "en": {
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
            """,
        "judge_system_prompt": \
            """
                You are an expert in data visualization and analytical communication.
                Your task is to review and evaluate code that produces charts or other visualizations,
                producing a detailed critique based on appropriateness, clarity, data-fidelity, aesthetics, technical correctness and improvement suggestions.
            """,
        "judge_user_prompt": \
            """
                The following code was used to create visualizations of a dataset.
                Please evaluate the visualizations according to the criteria and provide a detailed critique.
                Generated code:
                {code}
            """,
        "refactor_system_prompt": \
            """
                You are an experienced developer and code-refactoring agent.
                You receive judgments from a Judge agent and should refactor the affected code accordingly.
                Return the full, refactored code and briefly comment changes.
            """,
        "refactor_user_prompt": \
            """
                Here is the current code to be refactored:

                --- CODE START ---
                {code}
                --- CODE END ---

                Here are the verdicts from the Judge agent:
                {judge_messages}

                Please refactor the code based on these verdicts and return the full source code.
            """
    }
}


def get_prompt(language: str, key: str, **fmt):
    return PROMPTS[language][key].format(**fmt)
