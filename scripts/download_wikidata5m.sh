'''
Run this script as `sh scripts/download_wikidata5m.sh $DATA_DIR`
'''

cd $1
echo "saving data to $1..."
wget https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz
wget https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz
tar -xvzf wikidata5m_transductive.tar.gz
tar -xvzf wikidata5m_alias.tar.gz