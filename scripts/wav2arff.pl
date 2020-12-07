#!/usr/bin/perl

# Convert the .wav files in IEMOCAP to .arff files, one for each .wav file.
# This script works on enmcomp8,9,10 only (it needs openSmile)
# Example usage:
#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session1 -tdir arff/IS09_emotion -c config/IS09_emotion.conf
#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session2 -tdir arff/IS09_emotion -c config/IS09_emotion.conf
#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session3 -tdir arff/IS09_emotion -c config/IS09_emotion.conf
#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session4 -tdir arff/IS09_emotion -c config/IS09_emotion.conf
#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session5 -tdir arff/IS09_emotion -c config/IS09_emotion.conf

#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session1 -tdir arff/IS11_speaker_state -c config/IS11_speaker_state.conf
#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session2 -tdir arff/IS11_speaker_state -c config/IS11_speaker_state.conf
#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session3 -tdir arff/IS11_speaker_state -c config/IS11_speaker_state.conf
#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session4 -tdir arff/IS11_speaker_state -c config/IS11_speaker_state.conf
#   scripts/wav2arff.pl -sdir /corpus/iemocap/Session5 -tdir arff/IS11_speaker_state -c config/IS11_speaker_state.conf


use strict;
use warnings;
use Getopt::Long;
use File::Basename;

###################################################################################
# Define command line arguments and constant
###################################################################################
my $help;
my $srcdir = "/corpus/iemocap/Session1";
my $tgtdir = "arff/IS11_speaker_state";
my $config = "config/IS11_speaker_state.conf";

###################################################################################
#   prints usage if no command line parameters are passed or there is an unknown
#   parameter or help option is passed
###################################################################################
usage() if ($#ARGV < 1);
usage() if (!GetOptions('help|?' => \$help,
			'sdir=s' => \$srcdir,
			'tdir=s' => \$tgtdir,
			'c=s'    => \$config
			) or (defined $help));

sub usage
{
    print "Unknown option: @_\n" if ( @_ );
    print "usage: $0 [OPTION PARAMETER] ...\n";
    print "   Option Meaning                                Default\n";
    print "   -------------------------------------------------------------\n";
    print "   -sdir  Source directory                       $srcdir\n";
    print "   -tdir  Target directory                       $tgtdir\n";
    print "   -c     OpenSMILE Configuration file           $config\n";
    print "   --help Print this help page\n";
    exit;
}

my $opensmile = "/usr/local/openSMILE-2.1.0/SMILExtract";

##################################################################################################
# Get all filenames from source directory
##################################################################################################
my $srcfilelist = `find "$srcdir/sentences/wav/" -name "*.wav"`;
my @srcfiles = split('\s+',$srcfilelist);
`mkdir -p $tgtdir`;
foreach my $wavfile (@srcfiles) {
    my $bn = basename($wavfile,'.wav');
    my @field = split('/', dirname($wavfile), 4);
    my $tgtpath = "${tgtdir}/$field[3]";
    my $arfffile = "${tgtpath}/${bn}.arff";
    print "$wavfile --> $arfffile\n";
    `rm -f $arfffile`;
    `mkdir -p $tgtpath`;
    `$opensmile -C $config -I $wavfile -O $arfffile 2>/dev/null`;
}
exit;



############################################################
# Subroutines
############################################################
sub print_arrayofhash {
    my (%hashtbl) = @_;
    foreach my $k (keys %hashtbl) {
	print "$k ";
	my $size = @{$hashtbl{$k}};
	for (my $j=0; $j<$size-1; $j++) {
	    print "$hashtbl{$k}[$j],";
        }
	print "$hashtbl{$k}[$size-1]\n";
    }
}

sub print_hash {
    my (%hashtbl) = @_;
    foreach my $k (sort keys %hashtbl) {
	print "$k: $hashtbl{$k}\n";
    }
}


sub print_lst {
    my (%hashtbl) = @_;
    foreach my $k (sort keys %hashtbl) {
	print "$hashtbl{$k}\n";
    }
}
