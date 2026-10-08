name: notebooks-clean

on:
  pull_request:
  push:
    branches: [main]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install nbstripout
      - name: Verify notebooks carry no outputs
        shell: bash
        run: |
          shopt -s globstar nullglob
          files=(**/*.ipynb)
          [ ${#files[@]} -eq 0 ] && exit 0
          nbstripout --verify "${files[@]}"