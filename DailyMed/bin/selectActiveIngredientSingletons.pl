#!/usr/bin/env perl
$|=1;
use strict;

my $infile = "extracted/active_ingredients.txt";
my $outfile = "results/singleton_active_ingredients.txt";

my %seen;
open F, $infile;
while (<F>) {
	chomp;
	my($xml, $unii, $name) = split /\t/;
	$seen{$xml}{$unii} = $name;
}
close F;

open OF, ">$outfile";
while (my($xml, $stuff) = each %seen) {
	next unless $xml;
	next if scalar keys %$stuff > 1;
	print OF join("\t", $xml, %$stuff), "\n";
}
close OF;
