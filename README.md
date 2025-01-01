# PyPortfolio

This project aims to manage various types of share transactions of a specific portfolio (buys, sells, dividends, etc.) in a generic way through a `TransactionManager`. The manager allows registering multiple calculators (classes that implement the `BaseCalculator` interface), which calculate and populate specific columns for each transaction in the `DataFrame`.

## Table of Contents
1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Main Classes](#main-classes)
   - [TransactionManager](#transactionmanager)
   - [BaseCalculator](#basecalculator)
   - [FIFOCalculator](#fifocalculator)
   - [DividendCalculator](#dividendcalculator)
   - [AveragePriceCalculator](#averagepricecalculator)
6. [Example](#example)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

This library provides a modular approach for calculating and tracking financial transactions in a stock portfolio. The core class is `TransactionManager`, which:
- Takes a `DataFrame` containing transaction data.
- Allows registering multiple calculations for specific columns.
- Applies those calculations row by row, optionally converting the resulting column to a specified data type (`dtype`).

Included calculators:
1. **FIFOCalculator**: Implements [FIFO](https://en.wikipedia.org/wiki/FIFO) logic for sell transactions, consuming the oldest shares first. It calculates the resulting capital gain or loss based on the original purchase price.
2. **DividendCalculator**: Detects transactions of type _dividend_ and returns the dividend amount.
3. **AveragePriceCalculator**: Calculates the average share price after each buy transaction.

---

## File Structure

An example file structure might look like this (omitting some packaging details):

pyportfolio/ 
├── calculators/ 
│ ├── base_calculator.py 
│ ├── fifo_calculator.py 
│ ├── dividend_calculator.py 
│ └── average_price_calculator.py 
├── transaction_manager.py 
└── columns.py


Where:
- `transaction_manager.py` contains the `TransactionManager` class.
- `base_calculator.py` contains the abstract `BaseCalculator` class.
- `fifo_calculator.py` contains the `FIFOCalculator` class.
- `dividend_calculator.py` contains the `DividendCalculator` class.
- `average_price_calculator.py` contains the `AveragePriceCalculator` class.
- `columns.py` contains constants for column names and transaction types (for example, `TYPE_BUY`, `TYPE_SELL`, etc.).

---

## Installation

### Clone the repository:
   ```bash
   git clone https://github.com/username/pyportfolio.git
   ```
### (Optional) Create and activate a virtual environment:
```bash
Copiar código
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

###Install dependencies (if a requirements.txt file exists):
```bash
Copiar código
pip install -r requirements.txt
```

