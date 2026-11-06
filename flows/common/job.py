"""Orquestador: ejecuta las etapas del pipeline en orden."""
import argparse
import importlib

from flows.common.logging import get_logger

log = get_logger(__name__)

# Los módulos con nombre que empieza por dígito no se pueden usar con `import`;
# se cargan con importlib a partir de su ruta de paquete.
STAGES = {
    "ingestion": "flows.00_ingestion",
    #"silver": "flows.01_data_engineering",
    #"gold": "flows.03_aggregation",
    #"train": "flows.05_ml_models",
}


def run_stage(name: str):
    module = importlib.import_module(STAGES[name])
    log.info("Inicio de etapa: %s", name)
    result = module.run()
    log.info("Fin de etapa: %s", name)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = parser.parse_args(argv)
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        run_stage(name)


if __name__ == "__main__":
    main()