import os
import pytest
import pandas as pd
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectTableColumnCountToEqual,
    ExpectColumnValuesToBeBetween
)

def test_housing_data_quality():
    """Validates the raw housing dataset using Great Expectations v1.0+ syntax."""
    # 1. Path to your current data file
    data_path = os.path.join("data", "raw", "housing_data.csv")
    assert os.path.exists(data_path), f"Data file not found at {data_path}"
    
    # 2. Set up the modern Ephemeral Data Context
    context = gx.get_context(mode="ephemeral")
    
    # 3. Create the expectation suite using the v1.x standard
    suite = gx.ExpectationSuite(name="housing_data_suite")
    
    # Add data quality rules to the suite
    suite.add_expectation(ExpectColumnToExist(column="num_rooms"))
    suite.add_expectation(ExpectColumnToExist(column="square_feet"))
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="square_feet", min_value=100, max_value=10000)
    )
    
    # Save the suite to our ephemeral context
    context.suites.add(suite)
    
    # 4. Load data and validate it
    datasource = context.data_sources.add_pandas(name="my_datasource")
    asset = datasource.add_csv_asset(name="housing_asset", filepath_or_buffer=data_path)
    
    batch_definition = asset.add_batch_definition_whole_dataframe("all_data")
    batch = batch_definition.get_batch()
    
    validation_result = batch.validate(suite)
    
    # 5. Pytest assertion
    # 5. Clear Pytest assertion with a readable validation summary dump
    assert validation_result.success, (
        f"Data validation failed!\n"
        f"Statistics: {validation_result.statistics}\n"
        f"Detailed Results Summary:\n"
        f"{[res.to_json_dict() for res in validation_result.results if not res.success]}"
    )