/ Server-side bucketing of NYSE TAQ for the state-dependent Kyle's lambda study.
/ .
/ bucketquery is fully self-contained (no globals), so the whole lambda can be
/ sent to the TAQ server over IPC and only the per-minute table comes back.
/ .
/ Output: one row per (sym, bucket) for 09:30 to 15:59, with
/   mb      mid at the start of the bucket
/   dm      mid change over the bucket, in bps: 1e4*(mb[t+1]-mb[t])%mb[t]
/   sflow   Lee-Ready signed volume (shares)
/   vol     total volume (shares)
/   ntrd    trade count
/   spr     time-weighted relative spread (ask-bid)%mid, 1-second sampling
/   dep     time-weighted quoted depth bsize+asize, 1-second sampling (round lots on the course server)
/   nq      number of NBBO updates
/   ofi     Cont-Kukanov-Stoikov order-flow imbalance (round lots on the course server)
/   rv1     realized variance of 10-second log mid returns within the bucket
/   rv5     rv1 summed over the last 5 buckets (including this one)
/ .
/ Every column describes bucket t itself. Model features must be lagged one
/ bucket within each (sym, date), never across the overnight gap.

bucketquery:{[s;c;dt;w]
    / s: symbol list, c: trade condition chars, dt: date, w: bucket width (timespan)
    st:0D09:30; et:0D16:00;

    / Quotes: keep two-sided, uncrossed NBBO; include pre-open quotes so the
    / first trades and the 09:30:00 grid point have a prevailing quote.
    q:select time,sym,bid,ask,bsize,asize from nbbo
        where date=dt,sym in s,time<=et,bid>0,ask>bid;
    if[not `p=attr q`sym; q:update `p#sym from `sym`time xasc q];

    / Trades signed by Lee-Ready against the prevailing NBBO mid,
    / falling back to the tick rule at the mid or without a quote.
    tr:select time,sym,price,size from trade
        where date=dt,sym in s,time within (st;et-1),cond in c;
    tickrule:{d:"j"$signum 0f^x-prev x; 0^fills ?[d=0;0N;d]};
    tr:update tick:tickrule price by sym from tr;
    tr:update m:price^0.5*bid+ask from aj[`sym`time;tr;q];
    tr:update side:?[price>m;1;?[price<m;-1;-1+2*tick>0]] from tr;
    tb:select sflow:sum side*size,vol:sum size,ntrd:count i by sym,bucket:w xbar time from tr;

    / Prevailing quote on a 1-second grid: time-weighted spread and depth,
    / bucket-boundary mids, and 10-second returns for realized variance.
    secs:st+0D00:00:01*til 1+"j"$(et-st)%0D00:00:01;
    gq:aj[`sym`time;([] sym:s) cross ([] time:secs);q];
    gq:update mid:0.5*bid+ask from gq;
    gb:select spr:avg (ask-bid)%mid,dep:avg bsize+asize by sym,bucket:w xbar time
        from gq where time<et;
    bd:select sym,bucket:time,mb:mid from gq where time=w xbar time;
    bd:`sym`bucket xkey update dm:1e4*((next mb)-mb)%mb by sym from bd;
    r10:0!select last mid by sym,time:0D00:00:10 xbar time from gq;
    r10:select rv1:sum r*r by sym,bucket:w xbar time
        from update r:log mid%prev mid by sym from r10 where time<et;

    / Order-flow imbalance per NBBO update (Cont, Kukanov and Stoikov, 2014).
    / q evaluates right to left, so the four terms are grouped explicitly.
    qq:update pb:prev bid,pbs:prev bsize,pa:prev ask,pas:prev asize by sym from q;
    qq:update e:?[null pb;0f;"f"$(((bid>=pb)*bsize)-(bid<=pb)*pbs)-((ask<=pa)*asize)-(ask>=pa)*pas]
        from qq;
    qb:select nq:count i,ofi:sum e by sym,bucket:w xbar time from qq where time within (st;et-1);

    / Full sym x bucket grid so empty minutes appear with zero flow.
    bk:([] sym:s) cross ([] bucket:st+w*til "j"$(et-st)%w);
    bk:bk lj bd lj gb lj r10 lj tb lj qb;
    bk:update 0^sflow,0^vol,0^ntrd,0^nq,0^ofi,0f^rv1 from bk;
    bk:update rv5:5 msum rv1 by sym from bk;
    `date`sym`bucket xcols update date:dt from bk
    }
