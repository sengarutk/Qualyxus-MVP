#!/usr/bin/env python3
from src.training.train_yolo_seg import main, _parse_cli_overrides

if __name__ == "__main__":
    main(cli_overrides=_parse_cli_overrides())
