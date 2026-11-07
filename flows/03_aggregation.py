import pandas as pd
from typing import Dict

def group_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Genera cuatro tablas estadísticas agrupadas con groupby y devuelve un diccionario."""
    
    # 1. 'age_by_survival': Describe la edad agrupada por supervivencia
    age_by_survival = df.groupby('Survived')['Age'].describe().reset_index()
    
    # 2. 'survival_by_sex': Agrega media, suma y conteo de supervivencia por Sex
    survival_by_sex = df.groupby('Sex')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    
    # 3. 'survival_by_pclass': Igual que la anterior pero agrupada por Pclass
    survival_by_pclass = df.groupby('Pclass')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    
    # 4. 'survival_by_alone': Igual que la anterior pero agrupada por Alone
    survival_by_alone = df.groupby('Alone')['Survived'].agg(['mean', 'sum', 'count']).reset_index()
    
    # Retornamos el diccionario con todas las tablas agrupadas
    return {
        'age_by_survival': age_by_survival,
        'survival_by_sex': survival_by_sex,
        'survival_by_pclass': survival_by_pclass,
        'survival_by_alone': survival_by_alone
    }