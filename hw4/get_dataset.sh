# Download dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gNFhiaidM26gzXJCw1GQxeoAIeuK9X4N' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gNFhiaidM26gzXJCw1GQxeoAIeuK9X4N" -O hw4_data.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./hw4_data.zip

# Remove the downloaded zip file
rm ./hw4_data.zip
