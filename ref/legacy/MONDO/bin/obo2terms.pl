#!/bin/env perl
use strict;
require "../lib/libText.pl";

my($oboFile, $suppressFile, @topLevelIds) = @ARGV;
$oboFile ||= "data/mondo.obo";
$suppressFile ||= "data/suppress.txt";
@topLevelIds = qw/MONDO:0000001 MONDO:0021178 MONDO:0042489 HP:0000118 DOID:4/ unless @topLevelIds;
my $dz = "MONDO:0000001";
my %isa;
my %top;
my %keep;
my @types = qw/EXACT RELATED NARROW BROAD/;

my %suppress;
open SF, $suppressFile;
while (<SF>) {
	chomp;
	my($term, $curie) = split /\t/;
	$suppress{$term}{$curie || 'all'}++;
}
close SF;

my($name, $syn, $parents, $children, $means, $relevant, $obsolete, $allids) = parseObo($oboFile);

my %topNodes;
foreach my $id (sort keys %$name) {
	next if defined $parents->{$id};
	$name->{$id} =~ s/^disease //;
	$topNodes{$id} = $name->{$id};
	my @desc = keys %{$children->{$id}};
	while (@desc) {
		my $desc = shift @desc;
		$top{$desc}{$id}++;
		push @desc, keys %{$children->{$desc}};
	}
}

while (my($id, $name) = each %$name) {
	foreach my $root (@topLevelIds) {
		$keep{$id}++ if is_a($root, $id);
	}
	#print join("\t", $id, $name), "\n" unless $keep{$id};
}

#print scalar keys %keep, "\t", scalar keys %$name, "\n";

print join("\t", "name", @types, "precise", "imprecise", "term"), "\n";
foreach my $term (sort keys %$means) {
	my $doit;
	foreach my $id (keys %{$allids->{$term}}) {
		last if $doit = $keep{$id};
	}
	next unless $doit;
	my @names;
	@names = sort keys %{$means->{$term}{"name"}} if defined $means->{$term}{"name"};
	@names = sort keys %{$means->{$term}{"EXACT"}} if !@names && defined $means->{$term}{"EXACT"};
	next if $topNodes{$names[0]};
	my @display;
	foreach my $type ("name", @types) {
		if (defined $means->{$term}{$type}) {
			push @display, join(",", sort keys %{$means->{$term}{$type}});
		} else {
			push @display, 'NA';
		}
	}
	if (defined $relevant->{$term}) {
		push @display, scalar keys %{$relevant->{$term}[0]};
		push @display, scalar keys %{$relevant->{$term}[1]};
	} else {
		push @display, 0, 0;
	}
	push @display, $term;
	print join("\t", @display), "\n";
}

sub parseObo {
	my($file) = @_;
	my(%terms);

	my($id, %name, %isa, %syn, %children, %seen);
	my(%means, %relevant, %obsolete, %allids);
	open F, $file;
	while (<F>) {
		chomp;
		if (/^\[Term\]/) {
			$_ = <F>;
			chomp;
			if (/^id: (.+)/) {
				$id = $1;
				next;
			} else {
				die "unexpected $_, expecting term id\n";
			}
		} elsif (!$id) {
			next;
		}
		my($f, $v) = /^([^:]+): (.+)/;
		if ($f eq 'name') {
			if ($v =~ /^obsolete/) {
				$obsolete{$id}++;
				$id = '';
				next;
			}
			$name{$id} = $v;
			$means{$v}{'name'}{$id}++;
			$allids{$v}{$id}++;
			$relevant{$v}[0]{$id}++;
			#print join("\t", $id, $v, $seen{$v}), "\n" if $seen{$v};
			$seen{$v} = $id;
		} elsif ($f eq 'synonym') {
			my($synonym) = $v =~ /\"(.+?)\"/;
			next if $synonym eq $name{$id};
			next if $suppress{$synonym}{'all'} || $suppress{$synonym}{$id};
			foreach my $type (@types) {
				next unless $v =~ /$type/;
				$means{$synonym}{$type}{$id}++;
				$allids{$synonym}{$id}++;
				if ($type eq "EXACT") {
					$relevant{$synonym}[0]{$id}++;
				} else {
					$relevant{$synonym}[1]{$id}++;
				}
			}
			$syn{$id}{$synonym}++;
		} elsif ($f eq 'is_a') {
			my($parent) = $v =~ /^(\S+)/;
			die "no parent $id $f $v\n" unless $parent;
			$isa{$id}{$parent}++;
			$children{$parent}{$id}++;
		} elsif ($f eq 'is_obsolete') {
			$obsolete{$id}++;
			#$id = '';
		} elsif (!$f) {
			$id = '';
		}
	}
	close F;
	return \%name, \%syn, \%isa, \%children, \%means, \%relevant, \%obsolete, \%allids;
}

sub is_a {
	my($id, $query) = @_;
	my $prev = $isa{$id}{$query};
	return $prev if defined $prev;
	if ($id eq $query) {
		return $isa{$id}{$query} = 1;
	}
	foreach my $parent (keys %{$parents->{$query}}) {
		if (is_a($id, $parent)) {
			return $isa{$id}{$query} = 1;
		}
	}
	return $isa{$id}{$query} = 0;
}

