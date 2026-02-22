import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score


def cohen_kappa_agreement(eval_1, eval_2):
    """Calculated the cohen kappa between two sets of scores."""
    y1 = [
        eval_1.bugs.score,
        eval_1.transformation.score,
        eval_1.compliance.score,
        eval_1.type.score,
        eval_1.encoding.score,
        eval_1.aesthetics.score
    ]

    y2 = [
        eval_2.bugs.score,
        eval_2.transformation.score,
        eval_2.compliance.score,
        eval_2.type.score,
        eval_2.encoding.score,
        eval_2.aesthetics.score
    ]

    quad_kappa = cohen_kappa_score(y1, y2, weights='quadratic')

    return quad_kappa


def icc_agreement(evals):
    """Calculates the Intraclass Correlation Coefficient for more than 2 raters."""
    dimensions = ['bugs', 'transformation', 'compliance', 'type', 'encoding', 'aesthetics']

    data = pd.DataFrame({'item': dimensions})

    for i, e in enumerate(evals, start=1):
        data[f'rater_{i}'] = [
            e.bugs.score,
            e.transformation.score,
            e.compliance.score,
            e.type.score,
            e.encoding.score,
            e.aesthetics.score,
        ]

    df_long = data.melt(id_vars='item', var_name='rater', value_name='score')
    df_long['rater'] = df_long['rater'].astype('category')

    # calculate ICC (random raters, absolute agreement)
    icc = pg.intraclass_corr(
        data=df_long,
        targets='item',
        raters='rater',
        ratings='score'
    )

    # ICC(2,1): most used model
    icc_result = icc[icc['Type'] == 'ICC2'].reset_index(drop=True)

    return icc_result
