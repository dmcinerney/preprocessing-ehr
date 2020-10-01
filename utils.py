from tqdm import tqdm

def to_file(df, file, blocksize=1000):
    print("saving to "+file)
    pbar = tqdm(total=len(df))
    for i in range(0, len(df), blocksize):
        df[i:i+blocksize].to_csv(file, index=False, header=i==0, mode='a', compression='gzip')
        #df[i:i+blocksize].to_json(file, orient='records', lines=True, compression='gzip', date_format="iso", mode='a')
        pbar.update(n=len(df[i:i+blocksize]))
