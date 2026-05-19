import urllib.request

pages = ['http://127.0.0.1:5000/', 'http://127.0.0.1:5000/models']
for p in pages:
    try:
        data = urllib.request.urlopen(p, timeout=5).read().decode('utf-8')
        print(p, 'len=', len(data))
        print('Header match:', 'Profile Risk Assessment' in data)
        print('Risk label present:', 'Risk score' in data)
        print('Images present:', ('<img' in data) or ('static/plots' in data))
    except Exception as e:
        print('ERROR fetching', p, e)
        continue
    # write the fetched HTML for inspection
    out = 'tmp_' + p.replace('http://127.0.0.1:5000/','').replace('/','_') + '.html'
    open(out, 'w', encoding='utf-8').write(data)
    print('Wrote', out)
