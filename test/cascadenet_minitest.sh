cd demo
path=../config/test.json
time python training.py --path=$path
time python evaluating.py --path=$path
time python evaluating_timestamp_ratio.py --path=$path
time python evaluating-median_span.py --path=$path
time python evaluating-equal.py --path=$path
time python generate_copy.py --path=$path