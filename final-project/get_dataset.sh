# Download dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IYWPK8h9FWyo0p4-SCAatLGy0l5omQaw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IYWPK8h9FWyo0p4-SCAatLGy0l5omQaw" -O food_data.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./food_data.zip

# Remove the downloaded zip file
rm ./food_data.zip
