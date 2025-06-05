from pathlib import Path
import pandas as pd


from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
from google.protobuf.json_format import MessageToDict

def wandb2pandas(run_dirs: dict, metric_key='train/CrossEntropyLoss', metric_type=float, metric_name: str = 'loss', overwrite: bool = False):
    dfs = {}
    for run_id, run_dir in run_dirs.items():
        run_dir = Path(run_dir).resolve()
        assert run_dir.exists(), f"run_dir does not exist: {run_dir}"
        csv_path = run_dir / f"{metric_name}.csv"
        if csv_path.exists()and not overwrite:
            print(f"{csv_path} already exists, loading from csv")
            dfs[run_id] = pd.read_csv(csv_path)
            continue
        # Parse wandb files
        data = {'step': [], metric_name: []}
        for run_file in run_dir.glob("**/*.wandb"):
            print(f"parsing wandb file {run_file.as_posix()}")
            ds = datastore.DataStore()
            ds.open_for_scan(run_file)
            while True:
                record_data = ds.scan_data()
                if record_data is None: # eof
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(record_data)
                record_dict = MessageToDict(record, preserving_proto_field_name=True)
                # NOTE: we only care about 'history' records which contain logged metrics
                # {'num': '116', 'history': {'item': [{'nested_key': ['train/CrossEntropyLoss'], 'value_json': '11.284587860107422'}, {'nested_key': ['throughput/device/batches_per_second'], 'value_json': '1.3822355709055039'}, {'nested_key': ['optim/learning_rate_group0'], 'value_json': '0.00011066000000000001'}, {'nested_key': ['_timestamp'], 'value_json': '1.7296471903704662e+09'}, {'nested_key': ['optim/grad/transformer.blocks.0.ff_out.weight.norm'], 'value_json': '0.3185684382915497'}, {'nested_key': ['optim/grad/transformer.blocks.1.att_proj.weight.norm'], 'value_json': '0.24162694811820984'}, {'nested_key': ['optim/grad/transformer.blocks.1.ff_out.weight.norm'], 'value_json': '0.22841547429561615'}, {'nested_key': ['optim/total_grad_norm'], 'value_json': '1.0476723909378052'}, {'nested_key': ['optim/grad/transformer.blocks.0.ff_proj.weight.norm'], 'value_json': '0.3905589282512665'}, {'nested_key': ['throughput/total_training_Gflops'], 'value_json': '10239.202033664'}, {'nested_key': ['System/Peak GPU Memory (MB)'], 'value_json': '33574.60546875'}, {'nested_key': ['optim/learning_rate_group1'], 'value_json': '0.00011066000000000001'}, {'nested_key': ['train/Perplexity'], 'value_json': '79585.55297787744'}, {'nested_key': ['throughput/total_tokens'], 'value_json': '1048576'}, {'nested_key': ['_runtime'], 'value_json': '125.218246436'}, {'nested_key': ['optim/grad/transformer.wte.weight.norm'], 'value_json': '0.4280906319618225'}, {'nested_key': ['throughput/total_training_log_Gflops'], 'value_json': '9.233978969157079'}, {'nested_key': ['throughput/device/tokens_per_second'], 'value_json': '181172.3807497262'}, {'nested_key': ['_step'], 'value_json': '2'}, {'nested_key': ['optim/grad/transformer.blocks.0.att_proj.weight.norm'], 'value_json': '0.2968718707561493'}, {'nested_key': ['optim/grad/transformer.blocks.0.attn_out.weight.norm'], 'value_json': '0.4956618845462799'}, {'nested_key': ['optim/grad/transformer.blocks.1.attn_out.weight.norm'], 'value_json': '0.44308724999427795'}, {'nested_key': ['optim/grad/transformer.blocks.1.ff_proj.weight.norm'], 'value_json': '0.14013081789016724'}],'step': {'num': '2'}}}
                history = record_dict.get('history', None)
                if history is None:
                    continue
                step = history['step'].get('num', None)
                if step is None:
                    continue
                step = int(step)
                # metric = next(metric_type(x['value_json']) for x in history['item'] if x['nested_key'][0] == metric_key)
                try:
                    metric = next(metric_type(x['value_json']) for x in history['item'] if x['nested_key'][0] == metric_key)
                except StopIteration:
                    continue
                data['step'].append(step)
                data[metric_name].append(metric)
        # Convert to dataframe and save as csv
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        dfs[run_id] = df
    return dfs