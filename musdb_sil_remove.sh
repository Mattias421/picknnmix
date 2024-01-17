input_directory="/mnt/parscratch/users/acq22mc/data/musdb"
output_directory="/mnt/parscratch/users/acq22mc/data/musdb_no_sil"

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Find all files named "bass.wav" recursively
find "$input_directory" -type f -name "bass.wav" | while read -r file; do
    # Get the relative path of the file
    relative_path="/${file#$input_directory/}"

    # Replace slashes with underscores to create a unique output file name
    output_file="/$output_directory/${relative_path//\//_}.wav"

    ffmpeg -i "$file" -af silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-90dB -acodec pcm_s16le -ar 44100 -ac 2 -f wav "$output_file"

    echo "Processed: $file -> $output_file"
done

