import sys
import os
import json
import argparse
import pandas as pd

from ml_project.config import load_project_config, ConfigurationError
from ml_project.predictor import ProjectPredictor, IntegrityError
from ml_project.validation import InputValidationError

def main():
    parser = argparse.ArgumentParser(description="CLI відтворюваного ядра інференсу ML-моделі (ЛР №7)")
    parser.add_argument("--config", default="configs/project_config.json", help="Шлях до project_config.json")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # metadata
    subparsers.add_parser("metadata", help="Показати опис та метадані компонента")

    # validate
    val_p = subparsers.add_parser("validate", help="Валідувати вхідний JSON файл")
    val_p.add_argument("input_file", help="Шлях до input JSON")

    # predict
    pred_p = subparsers.add_parser("predict", help="Сформувати прогноз для одиничного запису JSON")
    pred_p.add_argument("input_file", help="Шлях до input JSON")

    # predict-batch
    batch_p = subparsers.add_parser("predict-batch", help="Сформувати прогноз для пакета з CSV чи JSON файлу")
    batch_p.add_argument("input_file", help="Шлях до input CSV або JSON")

    args = parser.parse_args()

    try:
        config = load_project_config(args.config)
        predictor = ProjectPredictor(config)

        if args.command == "metadata":
            print(json.dumps(predictor.describe(), indent=2, ensure_ascii=False))
            sys.exit(0)

        elif args.command == "validate":
            with open(args.input_file, "r", encoding="utf-8") as f:
                rec = json.load(f)
            vec, warns = predictor.validator.validate_record(rec)
            status = "accepted_with_warning" if warns else "accepted"
            print(json.dumps({"status": status, "features_count": len(vec), "warnings": warns}, indent=2, ensure_ascii=False))
            sys.exit(0)

        elif args.command == "predict":
            with open(args.input_file, "r", encoding="utf-8") as f:
                rec = json.load(f)
            res = predictor.predict_record(rec)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            sys.exit(0)

        elif args.command == "predict-batch":
            if args.input_file.endswith(".csv"):
                df = pd.read_csv(args.input_file)
                records = df.to_dict(orient="records")
            else:
                with open(args.input_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            res = predictor.predict_batch(records)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            sys.exit(0)

    except (ConfigurationError, IntegrityError, InputValidationError) as err:
        sys.stderr.write(f"ПОМИЛКА ВАЛІДАЦІЇ/КОНТРАКТУ: {err}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"СИСТЕМНА ПОМИЛКА: {exc}\n")
        sys.exit(2)

if __name__ == "__main__":
    main()