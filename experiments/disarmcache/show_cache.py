"""Print semantic-cache entries one by one for audit."""
import argparse, json, urllib.request

def main():
    p=argparse.ArgumentParser(); p.add_argument('--url',default='http://127.0.0.1:8080/cache'); p.add_argument('--limit',type=int,default=20); p.add_argument('--include-key',action='store_true'); a=p.parse_args()
    url=a.url + ('&' if '?' in a.url else '?') + 'include_key=1' if a.include_key else a.url
    with urllib.request.urlopen(url,timeout=30) as r: data=json.loads(r.read())
    print(f"缓存条目总数: {data.get('entry_count', 0)}")
    print(f"阈值: {data.get('threshold')}  键模式: {data.get('key_mode')}  格式: {data.get('key_format','internal')}")
    for i,e in enumerate(data.get('entries',[])[:a.limit],1):
        print(f"\n===== 缓存条目 {i} =====")
        for k in ('entry_id','query','answer','retriever_version','prompt_variant','poisoned_source','evidence_ids','created_at'):
            print(f"{k}: {e.get(k)}")
        docs=e.get('evidence_documents') or []
        print('知识库证据:')
        for j,d in enumerate(docs,1):
            print(f"  {j}. id={d.get('id')} poisoned={d.get('poisoned')} score={d.get('score')}")
            print(f"     {str(d.get('text','')).replace(chr(10),' ')[:300]}")
        if a.include_key: print('Q+D缓存键:\n'+str(e.get('cache_key_text','')))
if __name__ == '__main__': main()
