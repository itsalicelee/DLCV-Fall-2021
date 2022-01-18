# Download dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1E_X2X-91SVsv0u_b-57wOc_CZuxBKkvJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1E_X2X-91SVsv0u_b-57wOc_CZuxBKkvJ" -O hw1_data.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
mkdir hw1_data
unzip ./hw1_data.zip -d hw1_data

# Remove the downloaded zip file
rm ./hw1_data.zip
