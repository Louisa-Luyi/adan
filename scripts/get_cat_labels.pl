#!/usr/bin/perl

# Read the label file "label/emo_labels.txt". For each line in this file, 
# find the full path of corresponding .wav file. Print (1) the path (without
# the corpus directory and the .wav extension), (2) its
# categorial label, (3) Speaker ID, and (4) Gender in one line, separated by a space char.
# Example output:
#   Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_M008 fru M01 M  
# Usage:
#   scripts/get_cat_labels.pl | more > labels/emo_labels_cat.txt

use strict;
use warnings;
use Getopt::Long;
use File::Basename;

###################################################################################
# Define command line arguments and constant
###################################################################################
my $corpusdir = "/corpus/iemocap";
my $labfile = "../labels/emo_labels.txt";
my %idmap = (
    "01F" => "1",
    "02F" => "2",
    "03F" => "3",
    "04F" => "4",
    "05F" => "5",
    "01M" => "6",
    "02M" => "7",
    "03M" => "8",
    "04M" => "9",
    "05M" => "10"
    );


###################################################################################
#   Read "labels/emo_labels.txt" line-by-line.
###################################################################################
open(LABFILE, $labfile) || die "Can't open input file $labfile. $!";
while (my $line=<LABFILE>) {
    chomp($line);
    my ($interval,$turn,$emo_cat,$emo_dim) = split(/\t/,$line);
    my $file = `find $corpusdir -name ${turn}.wav`;
    chomp($file);
    $file =~ s/$corpusdir//;        # Remove /corpus/iemocap/ in the front
    $file =~ s/\.wav//;             # Remove .wav at the end
    my ($sess,$type,$utt) = split(/_/,$turn,3);
    $utt = substr $utt, -4;
    my $gender = substr $utt, 0, 1;
    $sess = substr $sess, 3, 2;
    my $spkid = $idmap{"${sess}${gender}"};
    print "$file $emo_cat $spkid $gender\n";
}
close(LABFILE) or die "Error in closing $labfile $!";
