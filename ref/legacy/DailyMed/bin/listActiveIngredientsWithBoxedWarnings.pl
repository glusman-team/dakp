#!/usr/bin/env perl
$|=1;
use strict;

my %ing;
my %unii;
my %xmls;
open ING, "extracted/active_ingredients.txt";
while (<ING>) {
	chomp;
	my($xml, $unii, $ing) = split /\t/;
	next if $unii eq 'NA';
	$ing{$unii} = $ing;
	$unii{$xml}{$unii}++;
	$unii{$ing} = $unii;
	$xmls{$unii}++;
}
close ING;

my %warn;
open BOX, "extracted/boxed_warnings.txt";
while (<BOX>) {
	my($xml, $text) = split /\t/;
	if (defined $unii{$xml}) {
		push @{$warn{$_}}, $xml foreach keys %{$unii{$xml}};
	#} else {
	#	print "#no ingredient for $xml\n";
	}
}
close BOX;


foreach my $unii (sort keys %warn) {
	print join("\t", scalar @{$warn{$unii}}, $xmls{$unii}, $unii, "\"$ing{$unii}\"", join(",", sort @{$warn{$unii}})), "\n";
}
