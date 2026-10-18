from moep.toolcall_eval import _schema, score_call

FN = [{"name": "calculate_triangle_area", "description": "",
       "parameters": {"type": "dict", "properties": {"base": {"type": "integer"}, "height": {"type": "integer"},
                                                     "unit": {"type": "string"}},
                      "required": ["base", "height"]}},
      {"name": "math.hypot", "description": "", "parameters": {"type": "dict", "properties": {"x": {"type": "float"}},
                                                               "required": ["x"]}}]
TRUTH = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]


def test_correct_call_with_optional_omitted():
    r = score_call({"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5.0}}, FN, TRUTH)
    assert r == {"valid": True, "correct": True}


def test_string_normalization_matches_bfcl():
    r = score_call({"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "Units."}},
                   FN, TRUTH)
    assert r["correct"]


def test_wrong_value_missing_required_and_extra_arg_are_incorrect_but_valid():
    for args in ({"base": 11, "height": 5}, {"base": 10}, {"base": 10, "height": 5, "colour": "red"}):
        assert score_call({"name": "calculate_triangle_area", "arguments": args}, FN, TRUTH) == \
            {"valid": True, "correct": False}


def test_unknown_function_or_no_call_is_invalid():
    assert not score_call({"name": "nope", "arguments": {}}, FN, TRUTH)["valid"]
    assert not score_call(None, FN, TRUTH)["valid"]


def test_schema_maps_bfcl_types():
    s = _schema({"type": "dict", "properties": {"a": {"type": "float"}, "b": {"type": "tuple", "items": {"type": "any"}}}})
    assert s == {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "array", "items": {}}}}
