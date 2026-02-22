import csv
import inspect
import os

import numpy as np
import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from data_science_agent.dtos.base import LidaEvaluationBase
from data_science_agent.dtos.wrapper.visualization import VisualizationWrapper
from data_science_agent.dtos.wrapper.SEVQ import SEVQ
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color, LLMMetadata
from data_science_agent.utils.enums import LLMModel, Color
from data_science_agent.utils import icc_agreement

prompt = Prompt(
    de={
        "system_prompt":
            """
            Du bist ein **kritischer, erfahrener Experte für Datenvisualisierung** und bewertest Code, 
            der Visualisierungen erstellt. Deine Aufgabe ist es, den Code **objektiv, streng und differenziert**
            danach zu bewerten, wie gut er das beschriebene Ziel der Visualisierung erfüllt.
    
            Gehe besonders kritisch vor – vermeide zu wohlwollende Beurteilungen.
            Beziehe dich auf nachvollziehbare Standards der Datenvisualisierung und logisch begründete Urteile.
            Wenn ein Aspekt unklar, fehlerhaft oder suboptimal ist, **bewerte ihn deutlich unter 8** und erkläre dies präzise.
    
            Verwende die folgenden sechs **LIDA-Bewertungsdimensionen**:
    
            1. **bugs** – Enthält der Code syntaktische oder logische Fehler, die die Ausführung verhindern oder das Ergebnis verfälschen?
               Falls der Code nicht lauffähig ist oder falsche Ergebnisse produziert, **bewerte zwingend unter 5**.
    
            2. **transformation** – Ist die Datenvorbereitung (Filtern, Aggregieren, Gruppieren etc.) korrekt, logisch und zweckmäßig für das Ziel?
               Eine nur teilweise sinnvolle Transformation darf **maximal eine mittlere Bewertung** erhalten.
    
            3. **compliance** – Wie gut erfüllt der Code das formulierte Visualisierungsziel und zeigt die intendierten Zusammenhänge?
               Wenn das Ziel nur teilweise erreicht wird, sollte der Wert im Bereich **4–7** liegen.
    
            4. **type** – Passt der gewählte Visualisierungstyp (z. B. Balken‑, Linien‑, Streudiagramm) wirklich zu den Daten und zur Fragestellung?
               Wenn der Typ missverständlich, unpassend oder unklar ist, bewerte **unter 6**.
    
            5. **encoding** – Werden Variablen korrekt und effektiv über visuelle Kanäle (Achsen, Farben, Formen, Größen) dargestellt?
               Fehlerhafte oder inkonsistente Achsenbeschriftungen, Farben oder Skalierungen führen zu einer **niedrigen Bewertung** (< 6).
    
            6. **aesthetics** – Ist das Erscheinungsbild (Layout, Farben, Lesbarkeit, Proportionen) ansprechend, klar und professionell?
               Schlechtes Layout, übermäßiger Text oder unklare Farben mindern die Punktzahl.
    
            Gib für **jede Dimension** eine Punktzahl von **1 (sehr schlecht)** bis **10 (ausgezeichnet)** 
            und **eine präzise, nachvollziehbare Begründung**, warum du genau diesen Wert gewählt hast.  
            Eine kurze Gesamteinschätzung am Ende (1–3 Sätze) ist erwünscht, aber **keine Gesamtnote** nötig.
            """,

        "user_prompt":
            """
            Bewerte den folgenden Python-Visualisierungscode in Hinblick auf das angegebene Ziel. 
            Verwende die sechs LIDA-Dimensionen (bugs, transformation, compliance, type, encoding, aesthetics).
            Sei kritisch und objektiv – vermeide übermäßig hohe Bewertungen, wenn der Code Schwächen hat oder unklar ist.
    
            VISUALISIERUNGSZIEL:
            {goal_description}
    
            Generierter Code:
            ```{programming_language}
            {code}
            ```
    
            Gib für jede Dimension eine Bewertung (1–10) mit klarer Begründung und schließe mit einer kurzen Gesamteinschätzung ab.
            """
    },

    en={
        "system_prompt":
            """
            You are a **critical, experienced data visualization expert** tasked with evaluating code that produces visualizations.
            Your evaluation should be **objective, rigorous, and differentiated**, focusing solely on how well the code fulfills the stated visualization goal.
    
            Be intentionally critical – do not give overly generous ratings.
            Base your judgments on clear visualization best practices and logical reasoning.
            If any aspect is unclear, flawed, or suboptimal, **score it clearly below 8** and explain precisely why.
    
            Use the following six **LIDA evaluation dimensions**:
    
            1. **bugs** – Does the code contain syntactic or logical errors that prevent execution or distort results?
               If the code is unusable or produces incorrect output, the score **must be below 5**.
    
            2. **transformation** – Is data preparation (filtering, aggregation, grouping, etc.) correct, logical, and appropriate for the goal?
               Superficial or partially correct transformations should receive **mid‑range or lower scores**.
    
            3. **compliance** – How well does the code achieve the visualization goal and reveal the intended relationships?
               Partial goal fulfillment should typically be scored between **4 and 7**.
    
            4. **type** – Is the chosen visualization type (bar, line, scatter, etc.) truly appropriate for the data and objective?
               If the chosen type is questionable or ambiguous, score **below 6**.
    
            5. **encoding** – Are variables correctly and effectively mapped to visual channels (axes, color, size, shape)?
               Misleading encodings, inconsistent scales, or confusing labels should result in a **low score (< 6)**.
    
            6. **aesthetics** – Is the overall appearance (layout, color palette, readability, proportions) professional, clear, and visually balanced?
               Poor layout, clutter, or low readability should lower the score.
    
            Provide a score **from 1 (poor)** to **10 (excellent)** for each dimension, 
            with a **well‑reasoned and specific justification** for each rating.  
            End with a concise overall remark (1–3 sentences), but no overall numeric rating.
            """,

        "user_prompt":
            """
            Evaluate the following Python visualization code with respect to the given goal.
            Reference the six LIDA dimensions (bugs, transformation, compliance, type, encoding, aesthetics).
            Be critical and objective — avoid inflated scores if the code has weaknesses or ambiguity.
    
            VISUALIZATION GOAL:
            {goal_description}
    
            Generated Code:
            ```{programming_language}
            {code}
            ```
    
            Provide a 1–10 rating with justification for each dimension, then a short overall summary at the end.
            """
    },
)

LidaEvaluation = import_language_dto(AGENT_LANGUAGE, LidaEvaluationBase)


@track_duration
def llm_evaluate_visualizations(state: AgentState) -> AgentState:
    """Evaluate the generated code for each visualization using LIDA evaluation metric."""
    system_prompt = prompt.get_prompt(AGENT_LANGUAGE, "system_prompt")
    programming_language = state["programming_language"].value
    output_path = os.path.join(state["output_path"], "sevq.csv")

    df = pd.DataFrame(
        columns=["vis_index", "model", 'bugs', 'transformation',
                 'compliance', 'type', 'encoding', 'aesthetics']
    )

    ira_df = pd.DataFrame(columns=["vis_index", "icc_type", "icc_value", "F", "pval", "ci95_lower", "ci95_upper"])

    for i, vis in enumerate(state["visualizations"]):
        vis: VisualizationWrapper

        gpt5 = _get_SEVQ(i, programming_language, system_prompt, vis, state, LLMModel.GPT_5)
        gpt4o = _get_SEVQ(i, programming_language, system_prompt, vis, state, LLMModel.GPT_4o)
        gemini = _get_SEVQ(i, programming_language, system_prompt, vis, state, LLMModel.GEMINI)
        grok = _get_SEVQ(i, programming_language, system_prompt, vis, state, LLMModel.GROK)
        claude4 = _get_SEVQ(i, programming_language, system_prompt, vis, state, LLMModel.CLAUDE_4)

        evaluation_results = [gpt5, gpt4o, gemini, grok, claude4]

        for e in evaluation_results:
            scores = e.lida_evaluation_score
            df.loc[len(df)] = [
                i,
                e.model,
                scores.bugs.score,
                scores.transformation.score,
                scores.compliance.score,
                scores.type.score,
                scores.encoding.score,
                scores.aesthetics.score,
            ]

        ira = icc_agreement([x.lida_evaluation_score for x in evaluation_results])
        print(ira)
        for _, r in ira.iterrows():
            lower, upper = float(r["CI95%"][0]), float(r["CI95%"][1])

            ira_df.loc[len(ira_df)] = [i, r["Type"], r["ICC"], r["F"], r["pval"], lower, upper]

        print(f"Visualisierung {i}: ICC={ira.iloc[0]['ICC']:.3f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    ira_df.to_csv(os.path.join(state["output_path"], "sevq_ira.csv"), index=False)

    state["is_before_refactoring"] = False

    return state


def _get_SEVQ(i, programming_language: str, system_prompt: str, vis: VisualizationWrapper, state: AgentState,
              model: LLMModel):
    temp_agent = create_agent(
        model=get_llm_model(model),
        system_prompt=SystemMessage(content=system_prompt),
        response_format=LidaEvaluation
    )

    user_prompt = prompt.get_prompt(
        AGENT_LANGUAGE,
        "user_prompt",
        goal_description=vis.goal,
        code=vis.code.code,
        programming_language=programming_language
    )

    llm_response = temp_agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
    lida_result: LidaEvaluation = llm_response["structured_response"]

    sevq: SEVQ = SEVQ(fig_index=i, model=model.value, lida_evaluation_score=lida_result)

    # wir printen genau das Gleiche in statistics
    print_color(f"LIDA Judge for vis#{i} with {model.value}", Color.WARNING)

    print_color(f"   - bugs: {lida_result.bugs.score} / 10", Color.OK_BLUE)
    print_color(f"   - transformation: {lida_result.transformation.score} / 10", Color.OK_BLUE)
    print_color(f"   - compliance: {lida_result.compliance.score} / 10", Color.OK_BLUE)
    print_color(f"   - type: {lida_result.type.score} / 10", Color.OK_BLUE)
    print_color(f"   - encoding: {lida_result.encoding.score} / 10", Color.OK_BLUE)
    print_color(f"   - aesthetics: {lida_result.aesthetics.score} / 10", Color.OK_BLUE)

    total_score = lida_result.bugs.score + lida_result.transformation.score + \
                  lida_result.compliance.score + lida_result.type.score + \
                  lida_result.encoding.score + lida_result.aesthetics.score
    print_color(f"\n  Total Score: {total_score:.2f}/60", Color.OK_GREEN)
    print_color(f"\n  Average Score: {round(float(total_score) / 6, 1)}/10", Color.OK_GREEN)

    for message in reversed(llm_response["messages"]):
        if isinstance(message, AIMessage):
            state["llm_metadata"].append(
                LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name))
            break
    return sevq
