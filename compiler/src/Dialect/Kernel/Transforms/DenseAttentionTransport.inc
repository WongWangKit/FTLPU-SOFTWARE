// Included in Linker. The native dense attention layouts scatter complete
// 32-element BF16 words; none of these address changes requires arithmetic
// or a transpose. Keep the ordinary move() path responsible for source
// ownership, passive-SR routing and its lane-preserving-word validation.
struct DensePhysicalElement { mlir::DictionaryAttr home; V shape; I index; };
DensePhysicalElement densePhysical(mlir::DictionaryAttr home, const V &shape, I index) {
  const auto native = [](mlir::DictionaryAttr p) { return static_cast<bool>(p.getAs<mlir::StringAttr>("dense_attention_role")); };
  const I word=s.target.memory().bytes_per_word, tile=s.target.throughput().mxm_rows;
  const I lanes=s.target.throughput().lanes_per_tile, tileRows=s.target.throughput().tile_rows;
    if (!native(home)) return DensePhysicalElement{home, shape, index};
    const auto role = text(home, "dense_attention_role");
    if(llvm::StringRef(role).starts_with("flash_")) {
      const I heads=field(home,"heads"),dim=field(home,"head_dim"),group=field(home,"query_heads_per_kv");
      const I qr=field(home,"flash_query_rows"),count=field(home,"instruction_count");
      if(shape.size()!=2 || shape[1]!=heads*dim || shape[0]!=32 || dim%32 || group<=0 || qr%32
          || index<0 || index>=product(shape))throw std::runtime_error("Flash tile logical geometry mismatch");
      const I token=index/shape[1],head=index%shape[1]/dim,d=index%dim;
      I row=0,lane=0,side=0;
      if(role=="flash_query" || role=="flash_context") {
        const auto headMajor=home.getAs<mlir::BoolAttr>("query_head_major");
        const I kv=head/group,query=headMajor && headMajor.getValue()
            ?(head%group)*32+token:token*group+head%group;
        side=kv%2;
        if(role=="flash_query") {row=(kv/2)*qr*(dim/32)+(d/32)*qr+query;lane=d%32;}
        else {row=((kv/2)*(qr/32)+query/32)*dim+(d/32)*32+query%32;lane=d%32;}
      } else if(role=="flash_key") {
        side=head%2;row=(head/2)*dim+(d/32)*32+token;lane=d%32;
      } else if(role=="flash_value") {
        side=1-head%2;row=(head/2)*dim+d;lane=token;
      } else throw std::runtime_error("unknown Flash tile physical role");
      auto ids=slices(home);
      if(ids.size()!=2 || row<0 || row>=count)throw std::runtime_error("Flash tile address exceeds planar allocation");
      auto planar=s.placement("fp16_pair_planar",ids,side?"west":"east",field(home,"bank"),field(home,"base_row"),count);
      return {planar,{count,word},row*word+lane};
    }
    const auto kind = text(home, "kind");
    const I heads = field(home, "heads"), dim = field(home, "head_dim");
    const I tokenStride = field(home, "token_stride");
    if (shape.size() != 2 || heads <= 0 || dim <= 0 || dim % word
        || shape[1] != heads * dim || index < 0 || index >= product(shape)
        || tokenStride < shape[0] || (role != "context" && tokenStride % tile))
      throw std::runtime_error("dense attention transport has incompatible logical geometry");
    const I token = index / shape[1], feature = index % shape[1];
    const I head = feature / dim, block = feature % dim / word;
    const I headBlocks = dim / word;
    const I blocksPerHalf = std::max<I>(1, headBlocks / 2);
    I row = 0, pair = 0, segmentReduction = block;
    std::string hemisphere;
    if (role == "context") {
      const auto packed = home.getAs<mlir::BoolAttr>("head_block_packed");
      const bool planar = packed && packed.getValue();
      const auto pairedAttr = home.getAs<mlir::BoolAttr>("hemisphere_paired");
      const bool paired = planar && pairedAttr && pairedAttr.getValue();
      if (planar ? (kind != "fp16_pair_planar" || slices(home).size()!=(paired?4u:2u))
                 : (kind != "fp16_head_planar" || slices(home).size() != size_t(2 * headBlocks)))
        throw std::runtime_error("dense attention context requires one planar byte pair per head block");
      row = (head * (planar ? headBlocks : 1) + (planar ? block : 0)) * tokenStride + token;
      pair = planar ? 0 : block;
      if (paired) {
        const I headsPerKv = field(home, "query_heads_per_kv");
        if (headsPerKv <= 0 || heads % headsPerKv)
          throw std::runtime_error("paired attention context lacks GQA ownership");
        pair = (head / headsPerKv) % 2;
      }
      const auto mirrored = home.getAs<mlir::BoolAttr>("mirrored");
      if (!mirrored || !mirrored.getValue())
        throw std::runtime_error("dense attention context ownership must be explicitly mirrored");
      hemisphere = "both";
    } else if (role == "query") {
      if (kind != "fp16_query_iw")
        throw std::runtime_error("dense attention query requires native IW layout");
      const I headsPerKv = field(home, "query_heads_per_kv");
      if (headsPerKv <= 0 || heads % headsPerKv)
        throw std::runtime_error("dense attention query is missing GQA ownership");
      hemisphere = (head / headsPerKv) % 2 ? "west" : "east";
      row = ((head * blocksPerHalf + block % blocksPerHalf)
                 * (tokenStride / tile) + token / tile) * tileRows
          + token % tile / lanes;
      pair = token % lanes;
    } else if (role == "key") {
      if (kind != "fp16_head_planar")
        throw std::runtime_error("dense attention key requires head-planar layout");
      hemisphere = head % 2 ? "west" : "east";
      row = (head * blocksPerHalf + block % blocksPerHalf) * tokenStride + token;
      pair = block / blocksPerHalf;
    } else if (role == "value") {
      if (kind != "fp16_value_x16" || slices(home).size() % (2 * lanes))
        throw std::runtime_error("dense attention value requires native distributed layout");
      hemisphere = head % 2 ? "west" : "east";
      const I groups = slices(home).size() / (2 * lanes);
      if (groups <= 0)
        throw std::runtime_error("dense attention value has no physical slice group");
      segmentReduction = block % groups;
      row = ((head * headBlocks + block) * (tokenStride / tile)
                 + token / tile) * tileRows + token % tile / lanes;
      pair = segmentReduction * lanes + token % lanes;
    } else {
      throw std::runtime_error("unknown dense attention transport role");
    }
    auto ids = slices(home);
    I bank = field(home, "bank"), base = field(home, "base_row");
    if (auto segments = home.getAs<mlir::ArrayAttr>("segments")) {
      bool found = false;
      for (auto attribute : segments) {
        auto segment = llvm::cast<mlir::DictionaryAttr>(attribute);
        if (segmentReduction < field(segment, "reduction_begin")
            || segmentReduction >= field(segment, "reduction_end")) continue;
        if (found) throw std::runtime_error("dense attention layout segments overlap");
        found = true;
        ids = slices(segment);
        bank = field(segment, "bank");
        base = field(segment, "base_row");
        if (role == "key") pair = 0;
        if (role == "value") pair = token % lanes;
      }
      if (!found)
        throw std::runtime_error("dense attention layout has no segment for a head block");
    } else if (role == "query" && block >= blocksPerHalf) {
      throw std::runtime_error("dense attention query requires explicit rotary-half segments");
    }
    const I count = field(home, "instruction_count");
    if (pair < 0 || 2 * pair + 1 >= I(ids.size()) || row < 0 || row >= count)
      throw std::runtime_error("dense attention transport exceeds its native allocation");
    auto planar = s.placement("fp16_pair_planar", {ids[2 * pair], ids[2 * pair + 1]},
        hemisphere, bank, base, count);
    return DensePhysicalElement{planar, {1, count * word}, row * word + feature % word};
}
// A standalone native core can receive immutable words at their final
// addresses. Split the compiler's C2C bindings, not the device program.
void bindDenseIngress(const Tensor &value, mlir::DictionaryAttr destination,
                      const std::string &name) {
  struct Group { mlir::DictionaryAttr home; std::map<I,V> words; };
  std::vector<Group> groups;
  const I word=s.target.memory().bytes_per_word;
  for(I index=0;index<value.size();++index) {
    auto mapped=densePhysical(destination,value.shape,index);
    auto found=std::find_if(groups.begin(),groups.end(),[&](const Group &g){return g.home==mapped.home;});
    if(found==groups.end()) {groups.push_back({mapped.home,{}});found=std::prev(groups.end());}
    auto [it,inserted]=found->words.try_emplace(mapped.index/word,V(word,-1));
    auto &lane=it->second[mapped.index%word];
    if(lane!=-1)throw std::runtime_error("attention ingress aliases logical elements");
    lane=index;
  }
  I piece=0;
  for(const auto &group:groups)for(auto it=group.words.begin();it!=group.words.end();) {
    const I first=it->first;I next=first;V indices;
    while(it!=group.words.end() && it->first==next) {
      if(llvm::is_contained(it->second,I{-1}))throw std::runtime_error("attention ingress contains an incomplete word");
      indices.insert(indices.end(),it->second.begin(),it->second.end());++it;++next;
    }
    auto tensor=view(value,indices,{1,I(indices.size())});
    auto home=s.change(group.home,J{{"base_row",field(group.home,"base_row")+first},
        {"instruction_count",next-first}});
    for(const auto &entry:live)
      if(overlap(home,entry.second) && field(home,"base_row")<field(entry.second,"base_row")+field(entry.second,"instruction_count")
          && field(entry.second,"base_row")<field(home,"base_row")+field(home,"instruction_count"))
        throw std::runtime_error("dense direct ingress overlaps live storage");
    const I id=inputs.size();inputs.push_back(tensor);
    s.function.insertArgument(s.function.getNumArguments(),type(&s.context,tensor),mlir::DictionaryAttr{},s.loc);
    s.binding(tensor,home,"input",id,name+".native"+std::to_string(piece++),s.function.getArgument(s.function.getNumArguments()-1));
    live[{-2,id}]=home;last[{-2,id}]=0;
  }
}
bool moveDenseAttention(mlir::DictionaryAttr src, mlir::DictionaryAttr dst,
    const V &sourceShape, const V &destShape, const V &ix, const V &pos) {
  const auto native = [](mlir::DictionaryAttr p) {
    const auto paired = p.getAs<mlir::BoolAttr>("hemisphere_paired");
    if (p.get("dense_attention_role") && text(p,"dense_attention_role") == "context" && text(p,"kind") == "fp16_pair_planar"
        && (!paired || !paired.getValue()))
      return false;
    return static_cast<bool>(p.getAs<mlir::StringAttr>("dense_attention_role"));
  };
  if (!native(src) && !native(dst)) return false;
  if (ix.size() != pos.size())
    throw std::runtime_error("dense attention transport has inconsistent view indices");
  struct CopyGroup {
    mlir::DictionaryAttr source, destination;
    V sourceShape, destinationShape, indices, positions;
  };
  std::vector<CopyGroup> groups;
  for (size_t i = 0; i < ix.size(); ++i) {
    auto from = densePhysical(src, sourceShape, ix[i]);
    auto to = densePhysical(dst, destShape, pos[i]);
    auto group = std::find_if(groups.begin(), groups.end(), [&](const CopyGroup &candidate) {
      return candidate.source == from.home && candidate.destination == to.home
          && candidate.sourceShape == from.shape && candidate.destinationShape == to.shape;
    });
    if (group == groups.end()) {
      groups.push_back({from.home, to.home, from.shape, to.shape, {}, {}});
      group = std::prev(groups.end());
    }
    group->indices.push_back(from.index);
    group->positions.push_back(to.index);
  }
  for (const auto &group : groups)
    move(group.source, group.destination, group.sourceShape, group.destinationShape,
        group.indices, group.positions);
  return true;
}
