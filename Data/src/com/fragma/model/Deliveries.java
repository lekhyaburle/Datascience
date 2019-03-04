package com.fragma.model;

public class Deliveries {
	private int match_id;
	private int inning;
	private String battingTeam;
	private String bowlingTeam;
	private int over;
	private int ball;
	private String batsman;
	private String bowler;
	private int wideruns;
	private int byeruns;
	private int legbyeruns;
	private int noballruns;
	private int penaltyruns;
	private int batsmanruns;
	private int extraruns;
	private int totalruns;
	public int getMatch_id() {
		return match_id;
	}
	public void setMatch_id(int match_id) {
		this.match_id = match_id;
	}
	public int getInning() {
		return inning;
	}
	public void setInning(int inning) {
		this.inning = inning;
	}
	public String getBattingTeam() {
		return battingTeam;
	}
	public void setBattingTeam(String battingTeam) {
		this.battingTeam = battingTeam;
	}
	public String getBowlingTeam() {
		return bowlingTeam;
	}
	public void setBowlingTeam(String bowlingTeam) {
		this.bowlingTeam = bowlingTeam;
	}
	public int getOver() {
		return over;
	}
	public void setOver(int over) {
		this.over = over;
	}
	public int getBall() {
		return ball;
	}
	public void setBall(int ball) {
		this.ball = ball;
	}
	public String getBatsman() {
		return batsman;
	}
	public void setBatsman(String batsman) {
		this.batsman = batsman;
	}
	public String getBowler() {
		return bowler;
	}
	public void setBowler(String bowler) {
		this.bowler = bowler;
	}
	public int getWideruns() {
		return wideruns;
	}
	public void setWideruns(int wideruns) {
		this.wideruns = wideruns;
	}
	public int getByeruns() {
		return byeruns;
	}
	public void setByeruns(int byeruns) {
		this.byeruns = byeruns;
	}
	public int getLegbyeruns() {
		return legbyeruns;
	}
	public void setLegbyeruns(int legbyeruns) {
		this.legbyeruns = legbyeruns;
	}
	public int getNoballruns() {
		return noballruns;
	}
	public void setNoballruns(int noballruns) {
		this.noballruns = noballruns;
	}
	public int getPenaltyruns() {
		return penaltyruns;
	}
	public void setPenaltyruns(int penaltyruns) {
		this.penaltyruns = penaltyruns;
	}
	public int getBatsmanruns() {
		return batsmanruns;
	}
	public void setBatsmanruns(int batsmanruns) {
		this.batsmanruns = batsmanruns;
	}
	public int getExtraruns() {
		return extraruns;
	}
	public void setExtraruns(int extraruns) {
		this.extraruns = extraruns;
	}
	public int getTotalruns() {
		return totalruns;
	}
	public void setTotalruns(int totalruns) {
		this.totalruns = totalruns;
	}
	
	public Deliveries() {
		super();
	}
	public Deliveries(int match_id, int inning, String battingTeam, String bowlingTeam, int over, int ball,
			String batsman, String bowler, int wideruns, int byeruns, int legbyeruns, int noballruns, int penaltyruns,
			int batsmanruns, int extraruns, int totalruns) {
		super();
		this.match_id = match_id;
		this.inning = inning;
		this.battingTeam = battingTeam;
		this.bowlingTeam = bowlingTeam;
		this.over = over;
		this.ball = ball;
		this.batsman = batsman;
		this.bowler = bowler;
		this.wideruns = wideruns;
		this.byeruns = byeruns;
		this.legbyeruns = legbyeruns;
		this.noballruns = noballruns;
		this.penaltyruns = penaltyruns;
		this.batsmanruns = batsmanruns;
		this.extraruns = extraruns;
		this.totalruns = totalruns;
	}
	
}
